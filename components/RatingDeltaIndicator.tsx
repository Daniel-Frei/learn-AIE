"use client";

type Props = {
  delta: number;
  label: string;
  testId?: string;
  className?: string;
};

function formatSignedDelta(delta: number): string {
  const rounded = Math.round(delta);
  const abs = Math.abs(rounded).toLocaleString("en-US");
  if (rounded > 0) return `+${abs}`;
  if (rounded < 0) return `-${abs}`;
  return "0";
}

export default function RatingDeltaIndicator({
  delta,
  label,
  testId,
  className = "",
}: Props) {
  const isPositive = delta >= 0;
  const direction = isPositive ? "increased" : "decreased";
  const formatted = formatSignedDelta(delta);
  const abs = Math.abs(Math.round(delta)).toLocaleString("en-US");

  return (
    <span
      data-testid={testId}
      aria-label={`${label} ${direction} by ${abs} points`}
      className={`inline-flex items-center gap-0.5 whitespace-nowrap text-[11px] font-medium ${
        isPositive ? "text-emerald-300/75" : "text-rose-300/75"
      } ${className}`}
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 12 12"
        className="h-3 w-3"
        fill="none"
      >
        <path
          d={
            isPositive
              ? "M6 10V2m0 0L2.5 5.5M6 2l3.5 3.5"
              : "M6 2v8m0 0L2.5 6.5M6 10l3.5-3.5"
          }
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.6"
        />
      </svg>
      <span>{formatted}</span>
    </span>
  );
}
