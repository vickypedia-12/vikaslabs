import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "vikaslabs",
  description: "Portfolio of Vickypedia",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <style>{`
          :root {
            --primary: hsl(217, 91%, 60%) !important;
            --primary-foreground: hsl(210, 11%, 4%) !important;
            --primary-glow: hsl(217, 91%, 70%) !important;
          }
        `}</style>
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
