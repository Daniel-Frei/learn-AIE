import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import MathText from "@/components/MathText";

describe("MathText", () => {
  it("renders prompt markdown tables with readable table markup", () => {
    const html = renderToStaticMarkup(
      <MathText
        inline
        className="text-lg font-semibold"
        text={`A latent-variable model uses these probabilities:

| Hidden state | Prior | Likelihood of \\(X=x\\) |
| --- | ---: | ---: |
| \\(Z=0\\) | 0.3 | 0.8 |
| \\(Z=1\\) | 0.7 | 0.2 |

What is \\(P(X=x)\\)?`}
      />,
    );

    expect(html).toContain("<table");
    expect(html).toContain("overflow-x-auto");
    expect(html).toContain("Hidden state");
    expect(html).toContain("Likelihood of");
    expect(html).toContain("katex");
  });
});
