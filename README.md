# Learning AI

Learning AI is a Next.js + TypeScript quiz app for studying artificial intelligence concepts.

## Public Repository Safety

This repository is intended to be safe to publish publicly. Keep real credentials in local or deployment environment variables only; do not commit `.env.local`, generated server logs, Supabase service-role keys, OpenAI keys, or participant-specific exports.

Required server-side secrets for deployed API behavior:

- `OPENAI_API_KEY` for detailed explanation chat.
- `SUPABASE_SERVICE_ROLE_KEY` for server-side shared quiz data routes.
- `QUESTION_REPORT_EXPORT_TOKEN` to enable production question-report export.

Client-exposed Supabase publishable keys are expected for web/mobile clients, but they are only safe with the committed Supabase RLS policies applied.

## Getting Started

First, install dependencies and run the development server:

```bash
make ci
make start
```

Open [http://localhost:43191](http://localhost:43191) with your browser to see the result.

## Windows Start Menu Launcher

On Windows, install a per-user Start Menu shortcut named `Learning AI` from a cloned checkout:

```powershell
make install-windows-start-menu
```

If you prefer npm for setup, run the equivalent script:

```powershell
npm run windows:start-menu:install
```

After installation, press Windows, search for `Learning AI`, and press Enter. The shortcut opens a PowerShell window in this repository and runs `make start`.
The installed shortcut expects `make` to be available on `PATH`.

To remove the shortcut:

```powershell
make uninstall-windows-start-menu
```

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
