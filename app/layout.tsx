import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";
import AppNav from "../components/AppNav";
import { reactSelectionPermissionErrorFilterScript } from "../lib/selectionPermissionErrorFilter";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Learning AI",
  description: "Quiz for learning artificial intelligence concepts",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} bg-slate-950 text-slate-50 antialiased`}
      >
        <script
          id="react-selection-permission-error-filter"
          dangerouslySetInnerHTML={{
            __html: reactSelectionPermissionErrorFilterScript,
          }}
        />
        <AppNav />
        {children}
      </body>
    </html>
  );
}
