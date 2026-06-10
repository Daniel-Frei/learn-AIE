"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/", label: "Questions" },
  { href: "/learn", label: "Learning" },
];

export default function AppNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/95 text-slate-50 backdrop-blur">
      <nav
        aria-label="Primary"
        className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3"
      >
        <Link href="/" className="text-sm font-semibold text-slate-100">
          Learning AI
        </Link>
        <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/70 p-1">
          {NAV_LINKS.map((link) => {
            const isActive =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);

            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={isActive ? "page" : undefined}
                className={`rounded-md px-3 py-2 text-sm font-semibold transition-colors ${
                  isActive
                    ? "bg-sky-400 text-slate-950"
                    : "text-slate-300 hover:bg-slate-800 hover:text-slate-50"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
