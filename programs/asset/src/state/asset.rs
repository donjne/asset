use bytemuck::{Pod, Zeroable};
use podded::{
    types::{PodBool, PodStr},
    ZeroCopy,
};
use shank::ShankType;
use solana_program::pubkey::Pubkey;

use super::{Delegate, Discriminator};
use crate::extensions::{Extension, ExtensionData, ExtensionType};

/// Maximum length of a name.
pub const MAX_NAME_LENGTH: usize = 32;

/// Maximum length of a symbol.
pub const MAX_SYMBOL_LENGTH: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct Asset {
    /// Account discriminator.
    pub discriminator: Discriminator,

    /// State of the asset.
    pub state: State,

    /// The PDA derivation bump.
    pub bump: u8,

    /// Indicates whether the asset is mutable.
    pub mutable: PodBool,

    /// Current holder of the asset.
    pub holder: Pubkey,

    /// Group of the asset.
    ///
    /// This is a reference to the asset that represents the group. When
    /// the asset is not part of a group, the group is zero'd out.
    pub group: Pubkey,

    /// Authority of the asset.
    ///
    /// The authority is the account that can update the metadata of the
    /// asset. This is typically the creator of the asset.
    pub authority: Pubkey,

    /// Delegate of the asset.
    ///
    /// The delegate is the account that can control the asset on behalf of
    /// the holder.
    pub delegate: Delegate,

    /// Name of the asset.
    pub name: PodStr<MAX_NAME_LENGTH>,

    /// Name of the asset.
    pub symbol: PodStr<MAX_SYMBOL_LENGTH>,

    // Padding to maintain 8-byte alignment.
    _padding: u8,
}

impl Asset {
    /// Length of the account data.
    pub const LEN: usize = std::mem::size_of::<Asset>();

    /// Seed used to derive the PDA.
    pub const SEED: &'static str = "asset";

    /// Returns the extension data of a given type.
    ///
    /// This function will return the first extension of the given type. If the
    /// extension is not found, `None` is returned.
    pub fn get<'a, T: ExtensionData<'a>>(data: &'a [u8]) -> Option<T> {
        let mut cursor = Asset::LEN;

        while cursor < data.len() {
            let extension = Extension::load(&data[cursor..cursor + Extension::LEN]);

            if extension.extension_type() == T::TYPE {
                return Some(T::from_bytes(&data[cursor + Extension::LEN..]));
            }

            cursor = extension.boundary() as usize;
        }

        None
    }

    /// Indicates whether the account contains an extension of a given type.
    pub fn contains(extension_type: ExtensionType, data: &[u8]) -> bool {
        let mut cursor = Asset::LEN;

        while cursor < data.len() {
            let extension = Extension::load(&data[cursor..cursor + Extension::LEN]);

            if extension.extension_type() == extension_type {
                return true;
            }

            cursor = extension.boundary() as usize;
        }

        false
    }

    /// Returns the extensions of the account.
    ///
    /// This function will return a list of `ExtensionType` that are present
    /// on the account.
    pub fn get_extensions(data: &[u8]) -> Vec<ExtensionType> {
        let mut cursor = Asset::LEN;
        let mut extensions = Vec::new();

        while cursor < data.len() {
            let extension = Extension::load(&data[cursor..cursor + Extension::LEN]);
            extensions.push(extension.extension_type());
            cursor = extension.boundary() as usize;
        }

        extensions
    }

    /// Returns the first extension of the account.
    ///
    /// This function will return a tuple containing the extension type and the
    /// offset of the extension data. If the account does not contain any extension,
    /// `None` is returned.
    pub fn first_extension(data: &[u8]) -> Option<(&Extension, usize)> {
        if Asset::LEN < data.len() {
            return Some((
                Extension::load(&data[Asset::LEN..]),
                Asset::LEN + Extension::LEN,
            ));
        }

        None
    }

    /// Returns the last extension of the account.
    ///
    /// This function will return a tuple containing the extension type and the
    /// offset of the extension data. If the account does not contain any extension,
    /// `None` is returned.
    pub fn last_extension(data: &[u8]) -> Option<(&Extension, usize)> {
        let mut cursor = Asset::LEN;
        let mut last = None;

        while cursor < data.len() {
            let extension = Extension::load(&data[cursor..]);
            last = Some((extension, cursor + Extension::LEN));
            cursor = extension.boundary() as usize;
        }

        last
    }
}

impl<'a> ZeroCopy<'a, Asset> for Asset {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, ShankType)]
pub enum State {
    #[default]
    Unlocked,
    Locked,
}

unsafe impl Pod for State {}

unsafe impl Zeroable for State {}
