const path = require("path");
const { generateIdl } = require("@metaplex-foundation/shank-js");

const idlDir = path.join(__dirname, "..", "idls");
const binaryInstallDir = path.join(__dirname, "..", ".crates");
const programDir = path.join(__dirname, "..", "programs");

generateIdl({
  generator: "shank",
  programName: "asset_program",
  programId: "AssetGtQBTSgm5s91d1RAQod5JmaZiJDxqsgtqrZud73",
  idlDir,
  binaryInstallDir,
  programDir: path.join(programDir, "asset"),
});