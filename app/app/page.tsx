import { ColorModeButton } from "@/components/ui/color-mode"
import { Box, Button, Container, Heading, Stack, Text } from "@chakra-ui/react"
import Link from "next/link"
import Image from "next/image"
import Logo from "@/public/logo.svg"

export default function Home() {
  return (
    <Box minH="100vh" position="relative">
      <Box position="absolute" top="4" right="4">
        <ColorModeButton />
      </Box>
      <Container maxW="container.md" py={20}>
        <Stack align="center" gap={10}>
          <Image src={Logo} alt="SurvHub Logo" width={160} height={160} />
          <Stack align="center" gap={4}>
            <Heading fontSize="6xl" fontWeight="extrabold" fontFamily="var(--font-outfit)" letterSpacing="tight">
              SurvHub
            </Heading>
            <Text fontSize="2xl" color="fg.muted" fontWeight="semibold" textAlign="center" fontFamily="var(--font-outfit)">
              A Survival Analysis Benchmark
            </Text>
          </Stack>
          <Button
            asChild
            colorPalette="blue"
            size="lg"
            borderRadius="full"
            px={8}
            fontFamily="var(--font-outfit)"
          >
            <Link href="/leaderboard">
              Show leaderboard
            </Link>
          </Button>
        </Stack>
      </Container>
    </Box>
  );
}
