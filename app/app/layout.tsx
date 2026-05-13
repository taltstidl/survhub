import { Provider } from "@/components/ui/provider"
import { Outfit } from "next/font/google"

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
})

export default function RootLayout(props: { children: React.ReactNode }) {
  const { children } = props
  return (
    <html suppressHydrationWarning className={outfit.variable}>
      <body>
        <Provider>{children}</Provider>
      </body>
    </html>
  )
}