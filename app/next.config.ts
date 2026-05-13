import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/survhub",
  images: {
    unoptimized: true,
  },
  experimental: {
    optimizePackageImports: ["@chakra-ui/react"],
  },
};

export default nextConfig;
