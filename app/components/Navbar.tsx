"use client"

import { Box, Flex, HStack, IconButton, Text, VStack, Button } from "@chakra-ui/react"
import Link from "next/link"
import Image from "next/image"
import { ColorModeButton } from "./ui/color-mode"
import { LuMenu, LuX } from "react-icons/lu"
import { useState } from "react"
import { usePathname } from "next/navigation"

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()

  const links = [
    { name: "Leaderboard", href: "/leaderboard" },
    { name: "Datasets", href: "/datasets" },
  ]

  return (
    <Box as="nav" position="sticky" top={0} zIndex={100} h={16}>
      <Box
        position="absolute"
        top={0}
        left={0}
        w="full"
        css={{ backdropFilter: "blur(20px)" }}
        bg="bg/50"
        borderBottomWidth="1px"
        borderColor="border.muted"
      >
        <HStack h={16} gap={2} alignItems="center" px={4} maxW="7xl" mx="auto">
          {/* Mobile Hamburger */}
          <IconButton
            display={{ base: "flex", md: "none" }}
            onClick={() => setIsOpen(!isOpen)}
            variant="ghost"
            aria-label="Open menu"
            size="sm"
            css={{
              _icon: {
                width: "5",
                height: "5",
              },
            }}
          >
            {isOpen ? <LuX /> : <LuMenu />}
          </IconButton>

          {/* Logo */}
          <Link href="/">
            <HStack gap={2}>
              <Image
                src="/logo.svg"
                alt="SurvHub Logo"
                width={32}
                height={32}
              />
              <Text fontWeight="bold" fontSize="xl" letterSpacing="tight">
                SurvHub
              </Text>
            </HStack>
          </Link>

          {/* Desktop Links */}
          <HStack gap={2} display={{ base: "none", md: "flex" }} ml={4}>
            {links.map((link) => {
              const isActive = pathname === link.href
              return (
                <Button
                  key={link.name}
                  asChild
                  variant="ghost"
                  fontSize="md"
                  color={isActive ? "fg" : "fg.muted"}
                >
                  <Link href={link.href}>
                    {link.name}
                  </Link>
                </Button>
              )
            })}
          </HStack>

          {/* Right side */}
          <Box ml="auto">
            <ColorModeButton />
          </Box>
        </HStack>

        {/* Mobile Menu */}
        <Box
          display={{ base: "block", md: "none" }}
          overflow="hidden"
          transition="all 0.3s ease-in-out"
          maxH={isOpen ? "400px" : "0px"}
          opacity={isOpen ? 1 : 0}
          visibility={isOpen ? "visible" : "hidden"}
        >
          <VStack gap={2} p={4} alignItems="stretch">
            {links.map((link) => {
              const isActive = pathname === link.href
              return (
                <Button
                  key={link.name}
                  asChild
                  variant="ghost"
                  w="full"
                  justifyContent="flex-start"
                  size="lg"
                  color={isActive ? "fg" : "fg.muted"}
                  onClick={() => setIsOpen(false)}
                >
                  <Link href={link.href}>
                    {link.name}
                  </Link>
                </Button>
              )
            })}
          </VStack>
        </Box>
      </Box>
    </Box>
  )
}
