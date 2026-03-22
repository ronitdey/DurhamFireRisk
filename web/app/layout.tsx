import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Durham Fire Risk — Duke East Campus",
  description:
    "Physics-based wildfire risk assessment for Duke University East Campus. " +
    "Interactive 3D visualization with per-building risk scoring, Rothermel fire spread simulation, and defensible space analysis.",
  openGraph: {
    title: "Durham Fire Risk Intelligence",
    description: "Interactive wildfire risk map for Duke East Campus",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <head>
        <link
          href="https://api.mapbox.com/mapbox-gl-js/v3.20.0/mapbox-gl.css"
          rel="stylesheet"
        />
      </head>
      <body className="h-full">{children}</body>
    </html>
  );
}
