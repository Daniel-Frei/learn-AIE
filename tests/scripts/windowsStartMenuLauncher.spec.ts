import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

function readRepoFile(relativePath: string) {
  return fs.readFileSync(path.join(process.cwd(), relativePath), "utf8");
}

describe("Windows Start Menu launcher contract", () => {
  it("installs a user-searchable Learning AI Start Menu shortcut", () => {
    const installer = readRepoFile("scripts/install-windows-start-menu.ps1");

    expect(installer).toContain('[string]$AppName = "Learning AI"');
    expect(installer).toContain("Microsoft\\Windows\\Start Menu\\Programs");
    expect(installer).toContain("start-learning-ai.ps1");
    expect(installer).toContain("-NoExit -ExecutionPolicy Bypass -File");
  });

  it("launches the app by running make start from the repository root", () => {
    const launcher = readRepoFile("scripts/start-learning-ai.ps1");

    expect(launcher).toContain("Set-Location -LiteralPath $repoRoot");
    expect(launcher).toContain("Refreshing local Next.js dev cache");
    expect(launcher).toContain(
      "Remove-Item -LiteralPath $resolvedNextDevCache -Recurse -Force",
    );
    expect(launcher).toContain("Command: make start");
    expect(launcher).toContain("& make start");
    expect(launcher.indexOf("Remove-Item")).toBeLessThan(
      launcher.indexOf("& make start"),
    );
  });
});
