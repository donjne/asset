//! This code was AUTOGENERATED using the kinobi library.
//! Please DO NOT EDIT THIS FILE, instead use visitors
//! to add features, then rerun kinobi to update it.
//!
//! [https://github.com/metaplex-foundation/kinobi]
//!

use borsh::BorshDeserialize;
use borsh::BorshSerialize;
use crate::generated::types::ExtensionType;

/// Accounts.
pub struct Initialize {
            /// Asset account (pda of `['asset', canvas pubkey]`)

    
              
          pub asset: solana_program::pubkey::Pubkey,
                /// Address to derive the PDA from

    
              
          pub canvas: solana_program::pubkey::Pubkey,
                /// The account paying for the storage fees

    
              
          pub payer: solana_program::pubkey::Pubkey,
                /// The system program

    
              
          pub system_program: solana_program::pubkey::Pubkey,
      }

impl Initialize {
  pub fn instruction(&self, args: InitializeInstructionArgs) -> solana_program::instruction::Instruction {
    self.instruction_with_remaining_accounts(args, &[])
  }
  #[allow(clippy::vec_init_then_push)]
  pub fn instruction_with_remaining_accounts(&self, args: InitializeInstructionArgs, remaining_accounts: &[solana_program::instruction::AccountMeta]) -> solana_program::instruction::Instruction {
    let mut accounts = Vec::with_capacity(4 + remaining_accounts.len());
                            accounts.push(solana_program::instruction::AccountMeta::new(
            self.asset,
            false
          ));
                                          accounts.push(solana_program::instruction::AccountMeta::new_readonly(
            self.canvas,
            true
          ));
                                          accounts.push(solana_program::instruction::AccountMeta::new(
            self.payer,
            true
          ));
                                          accounts.push(solana_program::instruction::AccountMeta::new_readonly(
            self.system_program,
            false
          ));
                      accounts.extend_from_slice(remaining_accounts);
    let mut data = InitializeInstructionData::new().try_to_vec().unwrap();
          let mut args = args.try_to_vec().unwrap();
      data.append(&mut args);
    
    solana_program::instruction::Instruction {
      program_id: crate::ASSET_ID,
      accounts,
      data,
    }
  }
}

#[derive(BorshDeserialize, BorshSerialize)]
struct InitializeInstructionData {
            discriminator: u8,
                        }

impl InitializeInstructionData {
  fn new() -> Self {
    Self {
                        discriminator: 1,
                                                            }
  }
}

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InitializeInstructionArgs {
                  pub extension_type: ExtensionType,
                pub length: u32,
                pub data: Option<Vec<u8>>,
      }


/// Instruction builder for `Initialize`.
///
/// ### Accounts:
///
                ///   0. `[writable]` asset
                ///   1. `[signer]` canvas
                      ///   2. `[writable, signer]` payer
                ///   3. `[optional]` system_program (default to `11111111111111111111111111111111`)
#[derive(Default)]
pub struct InitializeBuilder {
            asset: Option<solana_program::pubkey::Pubkey>,
                canvas: Option<solana_program::pubkey::Pubkey>,
                payer: Option<solana_program::pubkey::Pubkey>,
                system_program: Option<solana_program::pubkey::Pubkey>,
                        extension_type: Option<ExtensionType>,
                length: Option<u32>,
                data: Option<Vec<u8>>,
        __remaining_accounts: Vec<solana_program::instruction::AccountMeta>,
}

impl InitializeBuilder {
  pub fn new() -> Self {
    Self::default()
  }
            /// Asset account (pda of `['asset', canvas pubkey]`)
#[inline(always)]
    pub fn asset(&mut self, asset: solana_program::pubkey::Pubkey) -> &mut Self {
                        self.asset = Some(asset);
                    self
    }
            /// Address to derive the PDA from
#[inline(always)]
    pub fn canvas(&mut self, canvas: solana_program::pubkey::Pubkey) -> &mut Self {
                        self.canvas = Some(canvas);
                    self
    }
            /// The account paying for the storage fees
#[inline(always)]
    pub fn payer(&mut self, payer: solana_program::pubkey::Pubkey) -> &mut Self {
                        self.payer = Some(payer);
                    self
    }
            /// `[optional account, default to '11111111111111111111111111111111']`
/// The system program
#[inline(always)]
    pub fn system_program(&mut self, system_program: solana_program::pubkey::Pubkey) -> &mut Self {
                        self.system_program = Some(system_program);
                    self
    }
                    #[inline(always)]
      pub fn extension_type(&mut self, extension_type: ExtensionType) -> &mut Self {
        self.extension_type = Some(extension_type);
        self
      }
                #[inline(always)]
      pub fn length(&mut self, length: u32) -> &mut Self {
        self.length = Some(length);
        self
      }
                /// `[optional argument]`
#[inline(always)]
      pub fn data(&mut self, data: Vec<u8>) -> &mut Self {
        self.data = Some(data);
        self
      }
        /// Add an aditional account to the instruction.
  #[inline(always)]
  pub fn add_remaining_account(&mut self, account: solana_program::instruction::AccountMeta) -> &mut Self {
    self.__remaining_accounts.push(account);
    self
  }
  /// Add additional accounts to the instruction.
  #[inline(always)]
  pub fn add_remaining_accounts(&mut self, accounts: &[solana_program::instruction::AccountMeta]) -> &mut Self {
    self.__remaining_accounts.extend_from_slice(accounts);
    self
  }
  #[allow(clippy::clone_on_copy)]
  pub fn instruction(&self) -> solana_program::instruction::Instruction {
    let accounts = Initialize {
                              asset: self.asset.expect("asset is not set"),
                                        canvas: self.canvas.expect("canvas is not set"),
                                        payer: self.payer.expect("payer is not set"),
                                        system_program: self.system_program.unwrap_or(solana_program::pubkey!("11111111111111111111111111111111")),
                      };
          let args = InitializeInstructionArgs {
                                                              extension_type: self.extension_type.clone().expect("extension_type is not set"),
                                                                  length: self.length.clone().expect("length is not set"),
                                                                                  data: self.data.clone(),
                                                  };
    
    accounts.instruction_with_remaining_accounts(args, &self.__remaining_accounts)
  }
}

  /// `initialize` CPI accounts.
  pub struct InitializeCpiAccounts<'a, 'b> {
                  /// Asset account (pda of `['asset', canvas pubkey]`)

      
                    
              pub asset: &'b solana_program::account_info::AccountInfo<'a>,
                        /// Address to derive the PDA from

      
                    
              pub canvas: &'b solana_program::account_info::AccountInfo<'a>,
                        /// The account paying for the storage fees

      
                    
              pub payer: &'b solana_program::account_info::AccountInfo<'a>,
                        /// The system program

      
                    
              pub system_program: &'b solana_program::account_info::AccountInfo<'a>,
            }

/// `initialize` CPI instruction.
pub struct InitializeCpi<'a, 'b> {
  /// The program to invoke.
  pub __program: &'b solana_program::account_info::AccountInfo<'a>,
            /// Asset account (pda of `['asset', canvas pubkey]`)

    
              
          pub asset: &'b solana_program::account_info::AccountInfo<'a>,
                /// Address to derive the PDA from

    
              
          pub canvas: &'b solana_program::account_info::AccountInfo<'a>,
                /// The account paying for the storage fees

    
              
          pub payer: &'b solana_program::account_info::AccountInfo<'a>,
                /// The system program

    
              
          pub system_program: &'b solana_program::account_info::AccountInfo<'a>,
            /// The arguments for the instruction.
    pub __args: InitializeInstructionArgs,
  }

impl<'a, 'b> InitializeCpi<'a, 'b> {
  pub fn new(
    program: &'b solana_program::account_info::AccountInfo<'a>,
          accounts: InitializeCpiAccounts<'a, 'b>,
              args: InitializeInstructionArgs,
      ) -> Self {
    Self {
      __program: program,
              asset: accounts.asset,
              canvas: accounts.canvas,
              payer: accounts.payer,
              system_program: accounts.system_program,
                    __args: args,
          }
  }
  #[inline(always)]
  pub fn invoke(&self) -> solana_program::entrypoint::ProgramResult {
    self.invoke_signed_with_remaining_accounts(&[], &[])
  }
  #[inline(always)]
  pub fn invoke_with_remaining_accounts(&self, remaining_accounts: &[(&'b solana_program::account_info::AccountInfo<'a>, bool, bool)]) -> solana_program::entrypoint::ProgramResult {
    self.invoke_signed_with_remaining_accounts(&[], remaining_accounts)
  }
  #[inline(always)]
  pub fn invoke_signed(&self, signers_seeds: &[&[&[u8]]]) -> solana_program::entrypoint::ProgramResult {
    self.invoke_signed_with_remaining_accounts(signers_seeds, &[])
  }
  #[allow(clippy::clone_on_copy)]
  #[allow(clippy::vec_init_then_push)]
  pub fn invoke_signed_with_remaining_accounts(
    &self,
    signers_seeds: &[&[&[u8]]],
    remaining_accounts: &[(&'b solana_program::account_info::AccountInfo<'a>, bool, bool)]
  ) -> solana_program::entrypoint::ProgramResult {
    let mut accounts = Vec::with_capacity(4 + remaining_accounts.len());
                            accounts.push(solana_program::instruction::AccountMeta::new(
            *self.asset.key,
            false
          ));
                                          accounts.push(solana_program::instruction::AccountMeta::new_readonly(
            *self.canvas.key,
            true
          ));
                                          accounts.push(solana_program::instruction::AccountMeta::new(
            *self.payer.key,
            true
          ));
                                          accounts.push(solana_program::instruction::AccountMeta::new_readonly(
            *self.system_program.key,
            false
          ));
                      remaining_accounts.iter().for_each(|remaining_account| {
      accounts.push(solana_program::instruction::AccountMeta {
          pubkey: *remaining_account.0.key,
          is_signer: remaining_account.1,
          is_writable: remaining_account.2,
      })
    });
    let mut data = InitializeInstructionData::new().try_to_vec().unwrap();
          let mut args = self.__args.try_to_vec().unwrap();
      data.append(&mut args);
    
    let instruction = solana_program::instruction::Instruction {
      program_id: crate::ASSET_ID,
      accounts,
      data,
    };
    let mut account_infos = Vec::with_capacity(4 + 1 + remaining_accounts.len());
    account_infos.push(self.__program.clone());
                  account_infos.push(self.asset.clone());
                        account_infos.push(self.canvas.clone());
                        account_infos.push(self.payer.clone());
                        account_infos.push(self.system_program.clone());
              remaining_accounts.iter().for_each(|remaining_account| account_infos.push(remaining_account.0.clone()));

    if signers_seeds.is_empty() {
      solana_program::program::invoke(&instruction, &account_infos)
    } else {
      solana_program::program::invoke_signed(&instruction, &account_infos, signers_seeds)
    }
  }
}

/// Instruction builder for `Initialize` via CPI.
///
/// ### Accounts:
///
                ///   0. `[writable]` asset
                ///   1. `[signer]` canvas
                      ///   2. `[writable, signer]` payer
          ///   3. `[]` system_program
pub struct InitializeCpiBuilder<'a, 'b> {
  instruction: Box<InitializeCpiBuilderInstruction<'a, 'b>>,
}

impl<'a, 'b> InitializeCpiBuilder<'a, 'b> {
  pub fn new(program: &'b solana_program::account_info::AccountInfo<'a>) -> Self {
    let instruction = Box::new(InitializeCpiBuilderInstruction {
      __program: program,
              asset: None,
              canvas: None,
              payer: None,
              system_program: None,
                                            extension_type: None,
                                length: None,
                                data: None,
                    __remaining_accounts: Vec::new(),
    });
    Self { instruction }
  }
      /// Asset account (pda of `['asset', canvas pubkey]`)
#[inline(always)]
    pub fn asset(&mut self, asset: &'b solana_program::account_info::AccountInfo<'a>) -> &mut Self {
                        self.instruction.asset = Some(asset);
                    self
    }
      /// Address to derive the PDA from
#[inline(always)]
    pub fn canvas(&mut self, canvas: &'b solana_program::account_info::AccountInfo<'a>) -> &mut Self {
                        self.instruction.canvas = Some(canvas);
                    self
    }
      /// The account paying for the storage fees
#[inline(always)]
    pub fn payer(&mut self, payer: &'b solana_program::account_info::AccountInfo<'a>) -> &mut Self {
                        self.instruction.payer = Some(payer);
                    self
    }
      /// The system program
#[inline(always)]
    pub fn system_program(&mut self, system_program: &'b solana_program::account_info::AccountInfo<'a>) -> &mut Self {
                        self.instruction.system_program = Some(system_program);
                    self
    }
                    #[inline(always)]
      pub fn extension_type(&mut self, extension_type: ExtensionType) -> &mut Self {
        self.instruction.extension_type = Some(extension_type);
        self
      }
                #[inline(always)]
      pub fn length(&mut self, length: u32) -> &mut Self {
        self.instruction.length = Some(length);
        self
      }
                /// `[optional argument]`
#[inline(always)]
      pub fn data(&mut self, data: Vec<u8>) -> &mut Self {
        self.instruction.data = Some(data);
        self
      }
        /// Add an additional account to the instruction.
  #[inline(always)]
  pub fn add_remaining_account(&mut self, account: &'b solana_program::account_info::AccountInfo<'a>, is_writable: bool, is_signer: bool) -> &mut Self {
    self.instruction.__remaining_accounts.push((account, is_writable, is_signer));
    self
  }
  /// Add additional accounts to the instruction.
  ///
  /// Each account is represented by a tuple of the `AccountInfo`, a `bool` indicating whether the account is writable or not,
  /// and a `bool` indicating whether the account is a signer or not.
  #[inline(always)]
  pub fn add_remaining_accounts(&mut self, accounts: &[(&'b solana_program::account_info::AccountInfo<'a>, bool, bool)]) -> &mut Self {
    self.instruction.__remaining_accounts.extend_from_slice(accounts);
    self
  }
  #[inline(always)]
  pub fn invoke(&self) -> solana_program::entrypoint::ProgramResult {
    self.invoke_signed(&[])
  }
  #[allow(clippy::clone_on_copy)]
  #[allow(clippy::vec_init_then_push)]
  pub fn invoke_signed(&self, signers_seeds: &[&[&[u8]]]) -> solana_program::entrypoint::ProgramResult {
          let args = InitializeInstructionArgs {
                                                              extension_type: self.instruction.extension_type.clone().expect("extension_type is not set"),
                                                                  length: self.instruction.length.clone().expect("length is not set"),
                                                                                  data: self.instruction.data.clone(),
                                                  };
        let instruction = InitializeCpi {
        __program: self.instruction.__program,
                  
          asset: self.instruction.asset.expect("asset is not set"),
                  
          canvas: self.instruction.canvas.expect("canvas is not set"),
                  
          payer: self.instruction.payer.expect("payer is not set"),
                  
          system_program: self.instruction.system_program.expect("system_program is not set"),
                          __args: args,
            };
    instruction.invoke_signed_with_remaining_accounts(signers_seeds, &self.instruction.__remaining_accounts)
  }
}

struct InitializeCpiBuilderInstruction<'a, 'b> {
  __program: &'b solana_program::account_info::AccountInfo<'a>,
            asset: Option<&'b solana_program::account_info::AccountInfo<'a>>,
                canvas: Option<&'b solana_program::account_info::AccountInfo<'a>>,
                payer: Option<&'b solana_program::account_info::AccountInfo<'a>>,
                system_program: Option<&'b solana_program::account_info::AccountInfo<'a>>,
                        extension_type: Option<ExtensionType>,
                length: Option<u32>,
                data: Option<Vec<u8>>,
        /// Additional instruction accounts `(AccountInfo, is_writable, is_signer)`.
  __remaining_accounts: Vec<(&'b solana_program::account_info::AccountInfo<'a>, bool, bool)>,
}
