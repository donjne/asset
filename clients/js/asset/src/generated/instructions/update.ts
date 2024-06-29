/**
 * This code was AUTOGENERATED using the kinobi library.
 * Please DO NOT EDIT THIS FILE, instead use visitors
 * to add features, then rerun kinobi to update it.
 *
 * @see https://github.com/kinobi-so/kinobi
 */

import {
  Context,
  Option,
  OptionOrNullable,
  Pda,
  PublicKey,
  Signer,
  TransactionBuilder,
  none,
  transactionBuilder,
} from '@metaplex-foundation/umi';
import {
  Serializer,
  bool,
  mapSerializer,
  option,
  string,
  struct,
  u8,
} from '@metaplex-foundation/umi/serializers';
import {
  ResolvedAccount,
  ResolvedAccountsWithIndices,
  getAccountMetasAndSigners,
} from '../shared';
import {
  ExtensionInput,
  ExtensionInputArgs,
  getExtensionInputSerializer,
} from '../types';

// Accounts.
export type UpdateInstructionAccounts = {
  /** Asset account */
  asset: PublicKey | Pda;
  /** The authority of the asset */
  authority?: Signer;
  /** Extension buffer (uninitialized asset) account */
  buffer?: PublicKey | Pda;
  /** The asset defining the group, if applicable */
  group?: PublicKey | Pda;
  /** The account paying for the storage fees */
  payer?: Signer;
  /** The system program */
  systemProgram?: PublicKey | Pda;
};

// Data.
export type UpdateInstructionData = {
  discriminator: number;
  name: Option<string>;
  mutable: Option<boolean>;
  extension: Option<ExtensionInput>;
};

export type UpdateInstructionDataArgs = {
  name?: OptionOrNullable<string>;
  mutable?: OptionOrNullable<boolean>;
  extension?: OptionOrNullable<ExtensionInputArgs>;
};

export function getUpdateInstructionDataSerializer(): Serializer<
  UpdateInstructionDataArgs,
  UpdateInstructionData
> {
  return mapSerializer<UpdateInstructionDataArgs, any, UpdateInstructionData>(
    struct<UpdateInstructionData>(
      [
        ['discriminator', u8()],
        ['name', option(string())],
        ['mutable', option(bool())],
        ['extension', option(getExtensionInputSerializer())],
      ],
      { description: 'UpdateInstructionData' }
    ),
    (value) => ({
      ...value,
      discriminator: 10,
      name: value.name ?? none(),
      mutable: value.mutable ?? none(),
      extension: value.extension ?? none(),
    })
  ) as Serializer<UpdateInstructionDataArgs, UpdateInstructionData>;
}

// Args.
export type UpdateInstructionArgs = UpdateInstructionDataArgs;

// Instruction.
export function update(
  context: Pick<Context, 'identity' | 'programs'>,
  input: UpdateInstructionAccounts & UpdateInstructionArgs
): TransactionBuilder {
  // Program ID.
  const programId = context.programs.getPublicKey(
    'asset',
    'AssetGtQBTSgm5s91d1RAQod5JmaZiJDxqsgtqrZud73'
  );

  // Accounts.
  const resolvedAccounts = {
    asset: {
      index: 0,
      isWritable: true as boolean,
      value: input.asset ?? null,
    },
    authority: {
      index: 1,
      isWritable: false as boolean,
      value: input.authority ?? null,
    },
    buffer: {
      index: 2,
      isWritable: true as boolean,
      value: input.buffer ?? null,
    },
    group: {
      index: 3,
      isWritable: false as boolean,
      value: input.group ?? null,
    },
    payer: {
      index: 4,
      isWritable: true as boolean,
      value: input.payer ?? null,
    },
    systemProgram: {
      index: 5,
      isWritable: false as boolean,
      value: input.systemProgram ?? null,
    },
  } satisfies ResolvedAccountsWithIndices;

  // Arguments.
  const resolvedArgs: UpdateInstructionArgs = { ...input };

  // Default values.
  if (!resolvedAccounts.authority.value) {
    resolvedAccounts.authority.value = context.identity;
  }
  if (!resolvedAccounts.systemProgram.value) {
    if (resolvedAccounts.payer.value) {
      resolvedAccounts.systemProgram.value = context.programs.getPublicKey(
        'systemProgram',
        '11111111111111111111111111111111'
      );
      resolvedAccounts.systemProgram.isWritable = false;
    }
  }

  // Accounts in order.
  const orderedAccounts: ResolvedAccount[] = Object.values(
    resolvedAccounts
  ).sort((a, b) => a.index - b.index);

  // Keys and Signers.
  const [keys, signers] = getAccountMetasAndSigners(
    orderedAccounts,
    'programId',
    programId
  );

  // Data.
  const data = getUpdateInstructionDataSerializer().serialize(
    resolvedArgs as UpdateInstructionDataArgs
  );

  // Bytes Created On Chain.
  const bytesCreatedOnChain = 0;

  return transactionBuilder([
    { instruction: { keys, programId, data }, signers, bytesCreatedOnChain },
  ]);
}
