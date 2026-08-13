//! SRCI chunk — embedded shader source names, contents, and compiler arguments.
//!
//! The structures in this module mirror the packed `DxilSourceInfo*` records
//! published in `DxilContainer.h`. Zlib-compressed source contents remain
//! compressed: the container parser validates and exposes their metadata and
//! bytes without making compression support a requirement of this `no_std`
//! crate.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

const MAIN_HEADER_SIZE: usize = 8;
const SECTION_HEADER_SIZE: usize = 8;
const ARGS_HEADER_SIZE: usize = 12;
const NAMES_HEADER_SIZE: usize = 10;
const NAME_ENTRY_HEADER_SIZE: usize = 16;
const CONTENTS_HEADER_SIZE: usize = 20;
const CONTENT_ENTRY_HEADER_SIZE: usize = 12;

fn read_u16(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn align4(value: usize) -> Option<usize> {
    value.checked_add(3).map(|value| value & !3)
}

fn zeros(data: &[u8]) -> bool {
    data.iter().all(|&byte| byte == 0)
}

fn terminated_utf8(data: &[u8]) -> Option<&str> {
    let (&terminator, text) = data.split_last()?;
    (terminator == 0)
        .then(|| core::str::from_utf8(text).ok())
        .flatten()
}

/// Published `DxilSourceInfoSectionType` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceInfoSectionType {
    /// Compressed or uncompressed source-file contents.
    SourceContents,
    /// Source-file names and corresponding content sizes.
    SourceNames,
    /// Compiler argument name/value pairs.
    Arguments,
    /// Additive section type not known by this version of the crate.
    Unknown(u16),
}

impl SourceInfoSectionType {
    /// Decode the raw section tag.
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        match value {
            0 => Self::SourceContents,
            1 => Self::SourceNames,
            2 => Self::Arguments,
            value => Self::Unknown(value),
        }
    }

    /// Encode the raw section tag.
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::SourceContents => 0,
            Self::SourceNames => 1,
            Self::Arguments => 2,
            Self::Unknown(value) => value,
        }
    }
}

/// Published `DxilSourceInfo_SourceContentsCompressType` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceCompression {
    /// Entry records are stored directly in the section.
    None,
    /// Entry records are stored as a zlib stream.
    Zlib,
}

impl SourceCompression {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Zlib),
            _ => None,
        }
    }

    const fn to_u16(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Zlib => 1,
        }
    }
}

/// One compiler option represented by two null-terminated strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceArgument<'a> {
    /// Option name without a leading `-` or `/`; empty for an input file.
    pub name: Cow<'a, str>,
    /// Option value; empty for a flag without an argument.
    pub value: Cow<'a, str>,
}

/// `DxilSourceInfo_Args` and its argument pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceArguments<'a> {
    /// Reserved flags. Conforming data stores zero.
    pub flags: u32,
    /// Ordered compiler argument pairs.
    pub arguments: Vec<SourceArgument<'a>>,
}

/// One `DxilSourceInfo_SourceNamesEntry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceName<'a> {
    /// Reserved flags. Conforming data stores zero.
    pub flags: u32,
    /// UTF-8 source-file name, excluding the wire null terminator.
    pub name: Cow<'a, str>,
    /// Size of the corresponding source contents, including its null terminator.
    pub content_size_in_bytes: u32,
}

/// `DxilSourceInfo_SourceNames` and its entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceNames<'a> {
    /// Reserved flags. Conforming data stores zero.
    pub flags: u32,
    /// Ordered source-file descriptors.
    pub entries: Vec<SourceName<'a>>,
}

/// One uncompressed `DxilSourceInfo_SourceContentsEntry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceContent<'a> {
    /// Reserved flags. Conforming data stores zero.
    pub flags: u32,
    /// UTF-8 source contents, excluding the wire null terminator.
    pub content: Cow<'a, str>,
}

/// Source-content entry storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceContentData<'a> {
    /// Parsed uncompressed content records.
    Uncompressed(Vec<SourceContent<'a>>),
    /// A zlib stream containing `count` records when decompressed.
    Zlib {
        /// Compressed entry bytes.
        data: Cow<'a, [u8]>,
        /// Number of uncompressed content records.
        count: u32,
        /// Size of the complete uncompressed entry stream.
        uncompressed_size_in_bytes: u32,
    },
}

/// `DxilSourceInfo_SourceContents` and its entry storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceContents<'a> {
    /// Header field retained verbatim. DXC currently emits zero and its reader
    /// does not use this field; the enclosing section owns the actual size.
    pub aligned_size_in_bytes: u32,
    /// Reserved flags. Conforming data stores zero.
    pub flags: u16,
    /// Parsed or compressed content records.
    pub data: SourceContentData<'a>,
}

/// Typed payload of one `DxilSourceInfoSection`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceInfoSectionData<'a> {
    /// Source-content section.
    SourceContents(SourceContents<'a>),
    /// Source-name section.
    SourceNames(SourceNames<'a>),
    /// Compiler-arguments section.
    Arguments(SourceArguments<'a>),
    /// Additive section whose payload is not yet published.
    Unknown {
        /// Raw section type.
        section_type: u16,
        /// Bytes after the section header, including any section-local padding.
        data: Cow<'a, [u8]>,
    },
}

impl SourceInfoSectionData<'_> {
    fn section_type(&self) -> SourceInfoSectionType {
        match self {
            Self::SourceContents(_) => SourceInfoSectionType::SourceContents,
            Self::SourceNames(_) => SourceInfoSectionType::SourceNames,
            Self::Arguments(_) => SourceInfoSectionType::Arguments,
            Self::Unknown { section_type, .. } => SourceInfoSectionType::Unknown(*section_type),
        }
    }
}

/// One source-info section and its reserved section flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInfoSection<'a> {
    /// Reserved flags. Conforming data stores zero.
    pub flags: u16,
    /// Section payload.
    pub data: SourceInfoSectionData<'a>,
}

/// Parsed `DxilSourceInfo` (`SRCI`) payload.
#[derive(Debug, Clone)]
pub struct SourceInfo<'a> {
    /// Reserved flags. Conforming data stores zero.
    pub flags: u16,
    /// Sequential source-info sections.
    pub sections: Vec<SourceInfoSection<'a>>,
    original_data: Option<Cow<'a, [u8]>>,
    original_canonical: Option<Vec<u8>>,
}

fn parse_arguments(data: &[u8]) -> Option<SourceArguments<'_>> {
    if data.len() < ARGS_HEADER_SIZE {
        return None;
    }
    let flags = read_u32(data, 0)?;
    if flags != 0 {
        return None;
    }
    let size = usize::try_from(read_u32(data, 4)?).ok()?;
    let count = usize::try_from(read_u32(data, 8)?).ok()?;
    let pair_end = ARGS_HEADER_SIZE.checked_add(size)?;
    let section_size = align4(SECTION_HEADER_SIZE.checked_add(pair_end)?)?;
    let body_size = section_size.checked_sub(SECTION_HEADER_SIZE)?;
    if body_size != data.len() || !zeros(data.get(pair_end..)?) {
        return None;
    }

    let pair_data = data.get(ARGS_HEADER_SIZE..pair_end)?;
    let mut position = 0usize;
    let mut arguments = Vec::new();
    for _ in 0..count {
        let name_end = pair_data
            .get(position..)?
            .iter()
            .position(|&byte| byte == 0)?;
        let name_end = position.checked_add(name_end)?;
        let name = core::str::from_utf8(pair_data.get(position..name_end)?).ok()?;
        position = name_end.checked_add(1)?;

        let value_end = pair_data
            .get(position..)?
            .iter()
            .position(|&byte| byte == 0)?;
        let value_end = position.checked_add(value_end)?;
        let value = core::str::from_utf8(pair_data.get(position..value_end)?).ok()?;
        position = value_end.checked_add(1)?;
        arguments.push(SourceArgument {
            name: Cow::Borrowed(name),
            value: Cow::Borrowed(value),
        });
    }
    if position != pair_data.len() {
        return None;
    }

    Some(SourceArguments { flags, arguments })
}

fn parse_names(data: &[u8]) -> Option<SourceNames<'_>> {
    if data.len() < NAMES_HEADER_SIZE {
        return None;
    }
    let flags = read_u32(data, 0)?;
    if flags != 0 {
        return None;
    }
    let count = usize::try_from(read_u32(data, 4)?).ok()?;
    let entries_size = usize::from(read_u16(data, 8)?);
    let entries_end = NAMES_HEADER_SIZE.checked_add(entries_size)?;
    let section_size = align4(SECTION_HEADER_SIZE.checked_add(entries_end)?)?;
    let body_size = section_size.checked_sub(SECTION_HEADER_SIZE)?;
    if body_size != data.len() || !zeros(data.get(entries_end..)?) {
        return None;
    }

    let entries_data = data.get(NAMES_HEADER_SIZE..entries_end)?;
    let mut position = 0usize;
    let mut entries = Vec::new();
    for _ in 0..count {
        let header_end = position.checked_add(NAME_ENTRY_HEADER_SIZE)?;
        if header_end > entries_data.len() {
            return None;
        }
        let aligned_size = usize::try_from(read_u32(entries_data, position)?).ok()?;
        let entry_flags = read_u32(entries_data, position + 4)?;
        let name_size = usize::try_from(read_u32(entries_data, position + 8)?).ok()?;
        let content_size_in_bytes = read_u32(entries_data, position + 12)?;
        let unaligned_size = NAME_ENTRY_HEADER_SIZE.checked_add(name_size)?;
        if entry_flags != 0
            || name_size == 0
            || aligned_size != align4(unaligned_size)?
            || position.checked_add(aligned_size)? > entries_data.len()
        {
            return None;
        }
        let name_start = header_end;
        let name_end = position.checked_add(unaligned_size)?;
        let name = terminated_utf8(entries_data.get(name_start..name_end)?)?;
        if !zeros(entries_data.get(name_end..position + aligned_size)?) {
            return None;
        }
        entries.push(SourceName {
            flags: entry_flags,
            name: Cow::Borrowed(name),
            content_size_in_bytes,
        });
        position += aligned_size;
    }
    if position != entries_data.len() {
        return None;
    }

    Some(SourceNames { flags, entries })
}

fn parse_content_entries(data: &[u8], count: usize) -> Option<Vec<SourceContent<'_>>> {
    let mut position = 0usize;
    let mut entries = Vec::new();
    for _ in 0..count {
        let header_end = position.checked_add(CONTENT_ENTRY_HEADER_SIZE)?;
        if header_end > data.len() {
            return None;
        }
        let aligned_size = usize::try_from(read_u32(data, position)?).ok()?;
        let flags = read_u32(data, position + 4)?;
        let content_size = usize::try_from(read_u32(data, position + 8)?).ok()?;
        let unaligned_size = CONTENT_ENTRY_HEADER_SIZE.checked_add(content_size)?;
        if flags != 0
            || content_size == 0
            || aligned_size != align4(unaligned_size)?
            || position.checked_add(aligned_size)? > data.len()
        {
            return None;
        }
        let content = terminated_utf8(data.get(header_end..position + unaligned_size)?)?;
        if !zeros(data.get(position + unaligned_size..position + aligned_size)?) {
            return None;
        }
        entries.push(SourceContent {
            flags,
            content: Cow::Borrowed(content),
        });
        position += aligned_size;
    }
    (position == data.len()).then_some(entries)
}

fn parse_contents(data: &[u8]) -> Option<SourceContents<'_>> {
    if data.len() < CONTENTS_HEADER_SIZE {
        return None;
    }
    let aligned_size_in_bytes = read_u32(data, 0)?;
    let flags = read_u16(data, 4)?;
    if flags != 0 {
        return None;
    }
    let compression = SourceCompression::from_u16(read_u16(data, 6)?)?;
    let entries_size = usize::try_from(read_u32(data, 8)?).ok()?;
    let uncompressed_size = read_u32(data, 12)?;
    let count = read_u32(data, 16)?;
    let entries_end = CONTENTS_HEADER_SIZE.checked_add(entries_size)?;
    let section_size = align4(SECTION_HEADER_SIZE.checked_add(entries_end)?)?;
    let body_size = section_size.checked_sub(SECTION_HEADER_SIZE)?;
    if body_size != data.len() || !zeros(data.get(entries_end..)?) {
        return None;
    }
    let entry_data = data.get(CONTENTS_HEADER_SIZE..entries_end)?;
    let parsed_data = match compression {
        SourceCompression::None => {
            if usize::try_from(uncompressed_size).ok()? != entries_size {
                return None;
            }
            SourceContentData::Uncompressed(parse_content_entries(
                entry_data,
                usize::try_from(count).ok()?,
            )?)
        }
        SourceCompression::Zlib => SourceContentData::Zlib {
            data: Cow::Borrowed(entry_data),
            count,
            uncompressed_size_in_bytes: uncompressed_size,
        },
    };

    Some(SourceContents {
        aligned_size_in_bytes,
        flags,
        data: parsed_data,
    })
}

fn parse_section(data: &[u8]) -> Option<SourceInfoSection<'_>> {
    if data.len() < SECTION_HEADER_SIZE {
        return None;
    }
    let flags = read_u16(data, 4)?;
    if flags != 0 {
        return None;
    }
    let section_type = SourceInfoSectionType::from_u16(read_u16(data, 6)?);
    let body = &data[SECTION_HEADER_SIZE..];
    let section_data = match section_type {
        SourceInfoSectionType::SourceContents => {
            SourceInfoSectionData::SourceContents(parse_contents(body)?)
        }
        SourceInfoSectionType::SourceNames => {
            SourceInfoSectionData::SourceNames(parse_names(body)?)
        }
        SourceInfoSectionType::Arguments => {
            SourceInfoSectionData::Arguments(parse_arguments(body)?)
        }
        SourceInfoSectionType::Unknown(section_type) => SourceInfoSectionData::Unknown {
            section_type,
            data: Cow::Borrowed(body),
        },
    };
    Some(SourceInfoSection {
        flags,
        data: section_data,
    })
}

/// Parse a complete published `DxilSourceInfo` payload.
#[must_use]
pub fn parse_srci(data: &[u8]) -> Option<SourceInfo<'_>> {
    if data.len() < MAIN_HEADER_SIZE || data.len() % 4 != 0 {
        return None;
    }
    let aligned_size = usize::try_from(read_u32(data, 0)?).ok()?;
    let flags = read_u16(data, 4)?;
    let section_count = usize::from(read_u16(data, 6)?);
    if aligned_size != data.len() || flags != 0 {
        return None;
    }

    let mut position = MAIN_HEADER_SIZE;
    let mut sections = Vec::new();
    for _ in 0..section_count {
        let size = usize::try_from(read_u32(data, position)?).ok()?;
        let end = position.checked_add(size)?;
        if size < SECTION_HEADER_SIZE || size % 4 != 0 || end > data.len() {
            return None;
        }
        sections.push(parse_section(data.get(position..end)?)?);
        position = end;
    }
    if position != data.len() {
        return None;
    }

    let mut result = SourceInfo {
        flags,
        sections,
        original_data: None,
        original_canonical: None,
    };
    result.original_canonical = Some(write_srci_canonical(&result));
    result.original_data = Some(Cow::Borrowed(data));
    Some(result)
}

fn push_padding(output: &mut Vec<u8>, base: usize) {
    while (output.len() - base) % 4 != 0 {
        output.push(0);
    }
}

fn write_arguments(arguments: &SourceArguments<'_>) -> Vec<u8> {
    assert_eq!(arguments.flags, 0, "SRCI argument flags must be zero");
    let mut pairs = Vec::new();
    for argument in &arguments.arguments {
        assert!(!argument.name.as_bytes().contains(&0));
        assert!(!argument.value.as_bytes().contains(&0));
        pairs.extend_from_slice(argument.name.as_bytes());
        pairs.push(0);
        pairs.extend_from_slice(argument.value.as_bytes());
        pairs.push(0);
    }

    let mut output = Vec::new();
    output.extend_from_slice(&arguments.flags.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(pairs.len())
            .expect("SRCI argument bytes exceed u32")
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u32::try_from(arguments.arguments.len())
            .expect("SRCI argument count exceeds u32")
            .to_le_bytes(),
    );
    output.extend_from_slice(&pairs);
    push_padding(&mut output, SECTION_HEADER_SIZE);
    output
}

fn write_names(names: &SourceNames<'_>) -> Vec<u8> {
    assert_eq!(names.flags, 0, "SRCI source-name flags must be zero");
    let mut entries = Vec::new();
    for entry in &names.entries {
        assert_eq!(entry.flags, 0, "SRCI source-name entry flags must be zero");
        assert!(!entry.name.as_bytes().contains(&0));
        let name_size = entry
            .name
            .len()
            .checked_add(1)
            .expect("SRCI source name size overflow");
        let aligned_size = align4(
            NAME_ENTRY_HEADER_SIZE
                .checked_add(name_size)
                .expect("SRCI source-name entry size overflow"),
        )
        .expect("SRCI source-name entry size overflow");
        entries.extend_from_slice(
            &u32::try_from(aligned_size)
                .expect("SRCI source-name entry exceeds u32")
                .to_le_bytes(),
        );
        entries.extend_from_slice(&entry.flags.to_le_bytes());
        entries.extend_from_slice(
            &u32::try_from(name_size)
                .expect("SRCI source name exceeds u32")
                .to_le_bytes(),
        );
        entries.extend_from_slice(&entry.content_size_in_bytes.to_le_bytes());
        entries.extend_from_slice(entry.name.as_bytes());
        entries.push(0);
        while entries.len() % 4 != 0 {
            entries.push(0);
        }
    }

    let mut output = Vec::new();
    output.extend_from_slice(&names.flags.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(names.entries.len())
            .expect("SRCI source-name count exceeds u32")
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u16::try_from(entries.len())
            .expect("SRCI source-name entry bytes exceed u16")
            .to_le_bytes(),
    );
    output.extend_from_slice(&entries);
    push_padding(&mut output, SECTION_HEADER_SIZE);
    output
}

fn write_content_entries(entries: &[SourceContent<'_>]) -> Vec<u8> {
    let mut output = Vec::new();
    for entry in entries {
        assert_eq!(entry.flags, 0, "SRCI source-content flags must be zero");
        assert!(!entry.content.as_bytes().contains(&0));
        let content_size = entry
            .content
            .len()
            .checked_add(1)
            .expect("SRCI source content size overflow");
        let aligned_size = align4(
            CONTENT_ENTRY_HEADER_SIZE
                .checked_add(content_size)
                .expect("SRCI source-content entry size overflow"),
        )
        .expect("SRCI source-content entry size overflow");
        output.extend_from_slice(
            &u32::try_from(aligned_size)
                .expect("SRCI source-content entry exceeds u32")
                .to_le_bytes(),
        );
        output.extend_from_slice(&entry.flags.to_le_bytes());
        output.extend_from_slice(
            &u32::try_from(content_size)
                .expect("SRCI source content exceeds u32")
                .to_le_bytes(),
        );
        output.extend_from_slice(entry.content.as_bytes());
        output.push(0);
        while output.len() % 4 != 0 {
            output.push(0);
        }
    }
    output
}

fn write_contents(contents: &SourceContents<'_>) -> Vec<u8> {
    assert_eq!(contents.flags, 0, "SRCI source-content flags must be zero");
    let (compression, entries, count, uncompressed_size) = match &contents.data {
        SourceContentData::Uncompressed(entries) => {
            let data = write_content_entries(entries);
            let size = u32::try_from(data.len()).expect("SRCI content entries exceed u32");
            (
                SourceCompression::None,
                data,
                u32::try_from(entries.len()).expect("SRCI content count exceeds u32"),
                size,
            )
        }
        SourceContentData::Zlib {
            data,
            count,
            uncompressed_size_in_bytes,
        } => (
            SourceCompression::Zlib,
            data.to_vec(),
            *count,
            *uncompressed_size_in_bytes,
        ),
    };

    let mut output = Vec::new();
    output.extend_from_slice(&contents.aligned_size_in_bytes.to_le_bytes());
    output.extend_from_slice(&contents.flags.to_le_bytes());
    output.extend_from_slice(&compression.to_u16().to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(entries.len())
            .expect("SRCI stored contents exceed u32")
            .to_le_bytes(),
    );
    output.extend_from_slice(&uncompressed_size.to_le_bytes());
    output.extend_from_slice(&count.to_le_bytes());
    output.extend_from_slice(&entries);
    push_padding(&mut output, SECTION_HEADER_SIZE);
    output
}

fn write_section(section: &SourceInfoSection<'_>, output: &mut Vec<u8>) {
    assert_eq!(section.flags, 0, "SRCI section flags must be zero");
    let start = output.len();
    output.extend_from_slice(&0u32.to_le_bytes());
    output.extend_from_slice(&section.flags.to_le_bytes());
    output.extend_from_slice(&section.data.section_type().to_u16().to_le_bytes());
    let body = match &section.data {
        SourceInfoSectionData::SourceContents(contents) => write_contents(contents),
        SourceInfoSectionData::SourceNames(names) => write_names(names),
        SourceInfoSectionData::Arguments(arguments) => write_arguments(arguments),
        SourceInfoSectionData::Unknown { data, .. } => data.to_vec(),
    };
    output.extend_from_slice(&body);
    while (output.len() - start) % 4 != 0 {
        output.push(0);
    }
    let size = u32::try_from(output.len() - start).expect("SRCI section exceeds u32");
    output[start..start + 4].copy_from_slice(&size.to_le_bytes());
}

fn write_srci_canonical(source_info: &SourceInfo<'_>) -> Vec<u8> {
    assert_eq!(source_info.flags, 0, "SRCI flags must be zero");
    let mut output = Vec::new();
    output.extend_from_slice(&0u32.to_le_bytes());
    output.extend_from_slice(&source_info.flags.to_le_bytes());
    output.extend_from_slice(
        &u16::try_from(source_info.sections.len())
            .expect("SRCI section count exceeds u16")
            .to_le_bytes(),
    );
    for section in &source_info.sections {
        write_section(section, &mut output);
    }
    let size = u32::try_from(output.len()).expect("SRCI payload exceeds u32");
    output[..4].copy_from_slice(&size.to_le_bytes());
    output
}

impl<'a> ChunkParser<'a> for SourceInfo<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_srci(data)
    }
}

impl ChunkWriter for SourceInfo<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"SRCI"
    }

    fn write_payload(&self) -> Vec<u8> {
        let canonical = write_srci_canonical(self);
        if self.original_canonical.as_deref() == Some(canonical.as_slice()) {
            if let Some(original) = &self.original_data {
                return original.to_vec();
            }
        }
        canonical
    }
}

impl fmt::Display for SourceInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// Shader Source Info: {} section(s)",
            self.sections.len()
        )?;
        for (index, section) in self.sections.iter().enumerate() {
            match &section.data {
                SourceInfoSectionData::SourceNames(names) => {
                    writeln!(
                        f,
                        "//   [{index}] SourceNames: {} file(s)",
                        names.entries.len()
                    )?;
                }
                SourceInfoSectionData::SourceContents(contents) => match &contents.data {
                    SourceContentData::Uncompressed(entries) => {
                        writeln!(
                            f,
                            "//   [{index}] SourceContents: {} uncompressed file(s)",
                            entries.len()
                        )?;
                    }
                    SourceContentData::Zlib { data, count, .. } => {
                        writeln!(
                            f,
                            "//   [{index}] SourceContents: {count} zlib file(s), {} bytes",
                            data.len()
                        )?;
                    }
                },
                SourceInfoSectionData::Arguments(arguments) => {
                    writeln!(
                        f,
                        "//   [{index}] Arguments: {} pair(s)",
                        arguments.arguments.len()
                    )?;
                }
                SourceInfoSectionData::Unknown { section_type, data } => {
                    writeln!(
                        f,
                        "//   [{index}] Unknown(0x{section_type:04X}): {} bytes",
                        data.len()
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn published_sections_roundtrip() {
        let source_info = SourceInfo {
            flags: 0,
            sections: vec![
                SourceInfoSection {
                    flags: 0,
                    data: SourceInfoSectionData::SourceNames(SourceNames {
                        flags: 0,
                        entries: vec![SourceName {
                            flags: 0,
                            name: Cow::Borrowed("shader.hlsl"),
                            content_size_in_bytes: 14,
                        }],
                    }),
                },
                SourceInfoSection {
                    flags: 0,
                    data: SourceInfoSectionData::SourceContents(SourceContents {
                        aligned_size_in_bytes: 0,
                        flags: 0,
                        data: SourceContentData::Uncompressed(vec![SourceContent {
                            flags: 0,
                            content: Cow::Borrowed("float4 main()"),
                        }]),
                    }),
                },
                SourceInfoSection {
                    flags: 0,
                    data: SourceInfoSectionData::Arguments(SourceArguments {
                        flags: 0,
                        arguments: vec![SourceArgument {
                            name: Cow::Borrowed("T"),
                            value: Cow::Borrowed("ps_6_0"),
                        }],
                    }),
                },
            ],
            original_data: None,
            original_canonical: None,
        };
        let bytes = source_info.write_payload();
        let parsed = parse_srci(&bytes).expect("valid SRCI");
        assert_eq!(parsed.write_payload(), bytes);
        assert_eq!(parsed.sections.len(), 3);
    }

    #[test]
    fn rejects_reserved_flags_and_bad_section_sizes() {
        let mut bytes = vec![8, 0, 0, 0, 0, 0, 0, 0];
        assert!(parse_srci(&bytes).is_some());
        bytes[4] = 1;
        assert!(parse_srci(&bytes).is_none());

        let bytes = [12, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0];
        assert!(parse_srci(&bytes).is_none());
    }
}
