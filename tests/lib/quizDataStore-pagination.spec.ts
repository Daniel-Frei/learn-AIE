import { describe, expect, it } from "vitest";
import { fetchAllRows } from "@/lib/server/quizDataStore";

type FakePage = { question_id: string };

function makePagedQuery(pages: FakePage[][]) {
  const ranges: Array<[number, number]> = [];
  let pageIndex = 0;

  return {
    ranges,
    query: {
      select() {
        return {
          order() {
            return {
              range(from: number, to: number) {
                ranges.push([from, to]);
                const data = pages[pageIndex] ?? [];
                pageIndex += 1;
                return Promise.resolve({ data, error: null });
              },
            };
          },
        };
      },
    },
  };
}

describe("fetchAllRows", () => {
  it("keeps paging until the source returns fewer than 1000 rows", async () => {
    const firstPage = Array.from({ length: 1000 }, (_, index) => ({
      question_id: `q-${index + 1}`,
    }));
    const secondPage = Array.from({ length: 283 }, (_, index) => ({
      question_id: `q-${index + 1001}`,
    }));
    const { query, ranges } = makePagedQuery([firstPage, secondPage]);

    const rows = await fetchAllRows<FakePage>(
      query,
      "question_id",
      "question_id",
    );

    expect(rows).toHaveLength(1283);
    expect(rows[0]?.question_id).toBe("q-1");
    expect(rows.at(-1)?.question_id).toBe("q-1283");
    expect(ranges).toEqual([
      [0, 999],
      [1000, 1999],
    ]);
  });
});
