use num_derive::FromPrimitive;
use solana_program::{
    decode_error::DecodeError,
    msg,
    program_error::{PrintProgramError, ProgramError},
};
use thiserror::Error;

#[derive(Error, Clone, Debug, Eq, PartialEq, FromPrimitive)]
pub enum AssetError {
    /// 0 - Asset already initialized
    #[error("Asset already initialized")]
    AlreadyInitialized,

    /// 1 - Invalid account length
    #[error("Invalid account length")]
    InvalidAccountLength,

    /// 2 - Incomplete extension data
    #[error("Incomplete extension data")]
    IncompleteExtensionData,

    /// 3 - Uninitialized account
    #[error("Uninitialized account")]
    Uninitialized,

    /// 4 - Extension not found
    #[error("Extension not found")]
    ExtensionNotFound,
}

impl PrintProgramError for AssetError {
    fn print<E>(&self) {
        msg!(&self.to_string());
    }
}

impl From<AssetError> for ProgramError {
    fn from(e: AssetError) -> Self {
        ProgramError::Custom(e as u32)
    }
}

impl<T> DecodeError<T> for AssetError {
    fn type_of() -> &'static str {
        "nifty::asset"
    }
}

#[macro_export]
macro_rules! err {
    ( $error:expr ) => {{
        Err($error.into())
    }};
}