import { describe, expect, it } from "vitest";
import { isReactSelectionPermissionError } from "../../lib/selectionPermissionErrorFilter";

describe("isReactSelectionPermissionError", () => {
  it("matches reported React selection permission errors", () => {
    expect(
      isReactSelectionPermissionError(
        'Permission denied to access property "__reactFiber$abc"',
        "Error\n    at getClosestInstanceFromNode",
      ),
    ).toBe(true);
    expect(
      isReactSelectionPermissionError(
        'Permission denied to access property "correspondingUseElement"',
        "Error\n    at getEventTarget\n    at findInstanceBlockingEvent",
      ),
    ).toBe(true);
  });

  it("does not match unrelated permission or app errors", () => {
    expect(
      isReactSelectionPermissionError(
        'Permission denied to access property "realAppProperty"',
        "Error\n    at appFunction",
      ),
    ).toBe(false);
    expect(
      isReactSelectionPermissionError(
        'Permission denied to access property "__reactFiber$abc"',
        "Error\n    at appFunction",
      ),
    ).toBe(false);
    expect(isReactSelectionPermissionError("Something else failed")).toBe(
      false,
    );
  });
});
