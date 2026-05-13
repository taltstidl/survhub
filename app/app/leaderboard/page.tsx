"use client"

import { useState } from "react"
import {
  Badge,
  Box,
  Container,
  Grid,
  GridItem,
  HStack,
  Table,
  Text,
  VStack,
  createListCollection,
  useToken,
} from "@chakra-ui/react"
import { Chart, useChart } from "@chakra-ui/charts"
import { Navbar } from "@/components/Navbar"
import { useColorModeValue } from "@/components/ui/color-mode"
import { RadioCardRoot, RadioCardItem } from "@/components/ui/radio-card"
import {
  SelectContent,
  SelectItem,
  SelectItemText,
  SelectLabel,
  SelectRoot,
  SelectTrigger,
  SelectValueText,
} from "@/components/ui/select"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ErrorBar, Rectangle } from "recharts"
import { LuArrowUpDown, LuArrowUp, LuArrowDown } from "react-icons/lu"
import leaderboardData from "../../../benchmark/leaderboard.json"

const scenarios = [
  {
    id: "risk-strat",
    title: "Risk Stratification",
    description: "Single time-independent risk score or survival time",
  },
  {
    id: "surv-curve",
    title: "Survival Curve Estimation",
    description: "Full time-dependent survival probability curves",
  },
]

const metricsMap = {
  "risk-strat": [
    { id: "harrell-c", title: "Harrell's C-Index" },
    { id: "uno-c", title: "Uno's C-Index" },
    { id: "cd-auc", title: "Cumulative/Dynamic AUC" },
  ],
  "surv-curve": [
    { id: "antolini-c", title: "Antolini's C-Index" },
    { id: "cd-auc", title: "Cumulative/Dynamic AUC" },
    { id: "integrated-brier", title: "Integrated Brier" },
  ],
}

const typeColors: Record<string, string> = {
  classic: "#00BD9D",
  deep: "#00BBF9",
  foundation: "#9B5DE5"
}

const legendPayload = [
  { value: 'Classic', id: 'classic', color: typeColors.classic },
  { value: 'Deep Learning', id: 'deep', color: typeColors.deep },
  { value: 'Foundation Model', id: 'foundation', color: typeColors.foundation },
]

const datasetSizes = createListCollection({
  items: [
    { label: "Small Datasets", value: "small", description: "Samples ≤ 1000 & Features ≤ 100" },
    { label: "Large Datasets", value: "large", description: "Samples > 1000 | Features > 1000" },
  ],
})

const leaderboardMetrics = createListCollection({
  items: [
    { label: "Elo Ranking", value: "elo", description: "based on wins and losses in simulated game" },
    { label: "Improvability", value: "improvability", description: "based on difference to best-performing model" },
  ],
})

export default function LeaderboardPage() {
  const [scenario, setScenario] = useState("risk-strat")
  const availableMetrics = metricsMap[scenario as keyof typeof metricsMap]
  const [metric, setMetric] = useState(availableMetrics[0].id)

  const [datasetSize, setDatasetSize] = useState<string[]>(["small"])
  const [leaderboardMetric, setLeaderboardMetric] = useState<string[]>(["elo"])

  const [sortColumn, setSortColumn] = useState<string>("elo")
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc")

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("desc")
    }
  }

  const handleScenarioChange = (newScenario: string) => {
    setScenario(newScenario)
    const newMetrics = metricsMap[newScenario as keyof typeof metricsMap]
    setMetric(newMetrics[0].id)
  }

  const size = datasetSize[0]
  const lMetric = leaderboardMetric[0]
  const currentScenarioData = (leaderboardData as Record<string, any>)[scenario] || {}

  const chartData = Object.entries(currentScenarioData)
    .map(([key, modelData]: [string, any]) => {
      const sizeData = modelData[size]
      if (!sizeData) return { model: modelData.name, type: modelData.type, score: 0, ci: null }

      const metricData = sizeData[metric]
      if (!metricData) return { model: modelData.name, type: modelData.type, score: 0, ci: null }

      const valueData = metricData[lMetric]
      if (!valueData) return { model: modelData.name, type: modelData.type, score: 0, ci: null }

      const relCi = valueData.ci ? [Math.max(0, valueData.value - valueData.ci[0]), Math.max(0, valueData.ci[1] - valueData.value)] : null

      return {
        model: modelData.name,
        type: modelData.type,
        score: valueData.value,
        ci: relCi,
      }
    })

  const tableData = Object.entries(currentScenarioData)
    .map(([key, modelData]: [string, any]) => {
      const sizeData = modelData[size]
      if (!sizeData) return null

      const metricData = sizeData[metric]
      if (!metricData) return null

      return {
        model: modelData.name,
        type: modelData.type,
        elo: metricData.elo,
        improvability: metricData.improvability,
        rank: metricData.rank,
        mrr: metricData.mrr,
      }
    })
    .filter((d): d is any => d !== null)
    .sort((a, b) => {
      let valA: any
      let valB: any

      if (sortColumn === 'model') {
        valA = a.model
        valB = b.model
      } else {
        valA = a[sortColumn]?.value ?? 0
        valB = b[sortColumn]?.value ?? 0
      }

      if (valA < valB) return sortDirection === "asc" ? -1 : 1
      if (valA > valB) return sortDirection === "asc" ? 1 : -1
      return 0
    })

  const chart = useChart({ data: chartData })

  const [gray100, gray900, white] = useToken('colors', ['gray.100', 'gray.900', 'white'])
  const errorColor = useColorModeValue(gray900, white)
  const cursorColor = useColorModeValue(gray100, gray900)

  return (
    <Box minH="100vh" display="flex" flexDirection="column">
      <Navbar />
      <Box as="main" py={4} flex="1">
        <Container maxW="7xl" h="full" px={4}>
          <Grid templateColumns={{ base: "1fr", lg: "350px 1fr" }} gap={8} alignItems="start">
            {/* Left Column: Controls */}
            <GridItem>
              <VStack gap={8} align="stretch">

                {/* Scenario Selection */}
                <Box>
                  <Text fontWeight="semibold" fontSize="lg" mb={4}>Scenario</Text>
                  <RadioCardRoot
                    value={scenario}
                    onValueChange={(e) => {
                      if (e.value !== null) handleScenarioChange(e.value)
                    }}
                  >
                    <VStack gap={3} align="stretch">
                      {scenarios.map((scen) => (
                        <RadioCardItem
                          key={scen.id}
                          value={scen.id}
                          label={scen.title}
                          description={scen.description}
                          disabled={scen.id === "surv-curve"}
                        />
                      ))}
                    </VStack>
                  </RadioCardRoot>
                </Box>

                {/* Metric Selection */}
                <Box>
                  <Text fontWeight="semibold" fontSize="lg" mb={4}>Metric</Text>
                  <RadioCardRoot
                    value={metric}
                    onValueChange={(e) => {
                      if (e.value !== null) setMetric(e.value)
                    }}
                  >
                    <VStack gap={3} align="stretch">
                      {availableMetrics.map((met) => (
                        <RadioCardItem
                          key={met.id}
                          value={met.id}
                          label={met.title}
                          disabled={met.id !== "harrell-c"}
                        />
                      ))}
                    </VStack>
                  </RadioCardRoot>
                </Box>

              </VStack>
            </GridItem>

            {/* Right Column: Chart */}
            <GridItem
              minH="500px"
              display="flex"
              flexDirection="column"
            >
              <HStack gap={4} mb={6} alignItems="flex-end">
                <Box flex="1">
                  <SelectRoot
                    collection={datasetSizes}
                    size="sm"
                    value={datasetSize}
                    onValueChange={(e) => setDatasetSize(e.value)}
                  >
                    <SelectLabel mb={1}>Dataset Size</SelectLabel>
                    <SelectTrigger>
                      <SelectValueText placeholder="Select size" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasetSizes.items.map((item) => (
                        <SelectItem item={item} key={item.value}>
                          <Box>
                            <SelectItemText>{item.label}</SelectItemText>
                            <Text fontSize="xs" color="fg.muted">{item.description}</Text>
                          </Box>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </SelectRoot>
                </Box>

                <Box flex="1" display={{ base: "none", md: "block" }}>
                  <SelectRoot
                    collection={leaderboardMetrics}
                    size="sm"
                    value={leaderboardMetric}
                    onValueChange={(e) => setLeaderboardMetric(e.value)}
                  >
                    <SelectLabel mb={1}>Leaderboard Metric</SelectLabel>
                    <SelectTrigger>
                      <SelectValueText placeholder="Select metric" />
                    </SelectTrigger>
                    <SelectContent>
                      {leaderboardMetrics.items.map((item) => (
                        <SelectItem item={item} key={item.value}>
                          <Box>
                            <SelectItemText>{item.label}</SelectItemText>
                            <Text fontSize="xs" color="fg.muted">{item.description}</Text>
                          </Box>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </SelectRoot>
                </Box>
              </HStack>

              <Box display={{ base: "none", md: "block" }} flex="1" w="full" h="400px" mb={24}>
                {chartData.length > 0 ? (
                  <>
                    <HStack justify="center" gap={6} mb={4}>
                      {legendPayload.map((item) => (
                        <HStack key={item.id} gap={2}>
                          <Box w={3} h={3} bg={item.color} borderRadius="sm" />
                          <Text fontSize="sm" color="fg.muted">{item.value}</Text>
                        </HStack>
                      ))}
                    </HStack>
                    <Chart.Root maxH="sm" chart={chart}>
                      <BarChart data={chart.data} responsive>
                        <CartesianGrid stroke={chart.color("border")} strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="model" angle={-45} textAnchor="end" interval={0} />
                        <YAxis axisLine={false} tickLine={false} domain={[0, 'auto']} />
                        <Tooltip
                          cursor={{ fill: cursorColor }}
                          animationDuration={100}
                          content={<Chart.Tooltip />}
                        />
                        <Bar
                          dataKey="score"
                          name={leaderboardMetrics.items.find(i => i.value === leaderboardMetric[0])?.label || "Score"}
                          radius={[4, 4, 0, 0]}
                          shape={(props) => {
                            const color = typeColors[props.payload!.type];
                            return (
                              <Rectangle
                                {...props}
                                fill={color}
                                fillOpacity={0.33}
                                stroke={color}
                                strokeWidth={1}
                              />
                            );
                          }}
                        >
                          <ErrorBar dataKey="ci" width={4} strokeWidth={2} stroke={errorColor} />
                        </Bar>
                      </BarChart>
                    </Chart.Root>
                  </>
                ) : (
                  <Text color="fg.muted" textAlign="center" mt={10}>
                    No data available for this configuration.
                  </Text>
                )}
              </Box>

              {/* Leaderboard Table */}
              {tableData.length > 0 && (
                <Table.Root size="sm" variant="outline" rounded="md" overflow="hidden">
                  <Table.Header>
                    <Table.Row bg="bg.subtle">
                      <Table.ColumnHeader
                        onClick={() => handleSort('model')}
                        cursor="pointer"
                        _hover={{ bg: "bg.muted" }}
                      >
                        <HStack gap={1}>
                          <Text>Model</Text>
                          {sortColumn === 'model' ? (
                            sortDirection === 'asc' ? <LuArrowUp /> : <LuArrowDown />
                          ) : (
                            <LuArrowUpDown style={{ opacity: 0.3 }} />
                          )}
                        </HStack>
                      </Table.ColumnHeader>
                      <Table.ColumnHeader
                        textAlign="right"
                        onClick={() => handleSort('elo')}
                        cursor="pointer"
                        _hover={{ bg: "bg.muted" }}
                      >
                        <HStack justify="flex-end" gap={1}>
                          <Text>Elo</Text>
                          {sortColumn === 'elo' ? (
                            sortDirection === 'asc' ? <LuArrowUp /> : <LuArrowDown />
                          ) : (
                            <LuArrowUpDown style={{ opacity: 0.3 }} />
                          )}
                        </HStack>
                      </Table.ColumnHeader>
                      <Table.ColumnHeader
                        textAlign="right"
                        onClick={() => handleSort('improvability')}
                        cursor="pointer"
                        _hover={{ bg: "bg.muted" }}
                      >
                        <HStack justify="flex-end" gap={1}>
                          <Text>Improvability</Text>
                          {sortColumn === 'improvability' ? (
                            sortDirection === 'asc' ? <LuArrowUp /> : <LuArrowDown />
                          ) : (
                            <LuArrowUpDown style={{ opacity: 0.3 }} />
                          )}
                        </HStack>
                      </Table.ColumnHeader>
                      <Table.ColumnHeader
                        textAlign="right"
                        onClick={() => handleSort('rank')}
                        cursor="pointer"
                        _hover={{ bg: "bg.muted" }}
                        display={{ base: "none", md: "table-cell" }}
                      >
                        <HStack justify="flex-end" gap={1}>
                          <Text>Average Rank</Text>
                          {sortColumn === 'rank' ? (
                            sortDirection === 'asc' ? <LuArrowUp /> : <LuArrowDown />
                          ) : (
                            <LuArrowUpDown style={{ opacity: 0.3 }} />
                          )}
                        </HStack>
                      </Table.ColumnHeader>
                      <Table.ColumnHeader
                        textAlign="right"
                        onClick={() => handleSort('mrr')}
                        cursor="pointer"
                        _hover={{ bg: "bg.muted" }}
                        display={{ base: "none", md: "table-cell" }}
                      >
                        <HStack justify="flex-end" gap={1}>
                          <Text>MRR</Text>
                          {sortColumn === 'mrr' ? (
                            sortDirection === 'asc' ? <LuArrowUp /> : <LuArrowDown />
                          ) : (
                            <LuArrowUpDown style={{ opacity: 0.3 }} />
                          )}
                        </HStack>
                      </Table.ColumnHeader>
                    </Table.Row>
                  </Table.Header>
                  <Table.Body>
                    {tableData.map((row: any) => (
                      <Table.Row key={row.model}>
                        <Table.Cell>
                          <VStack align="start" gap={1}>
                            <Text fontWeight="medium">{row.model}</Text>
                            <Badge
                              variant="solid"
                              size="xs"
                              css={{ backgroundColor: typeColors[row.type] }}
                            >
                              {row.type}
                            </Badge>
                          </VStack>
                        </Table.Cell>
                        <Table.Cell textAlign="right">
                          <VStack align="end" gap={0}>
                            <Text>{row.elo.value.toFixed(1)}</Text>
                            {row.elo.ci && (
                              <Text fontSize="xs" color="fg.muted">
                                [{row.elo.ci[0].toFixed(1)}, {row.elo.ci[1].toFixed(1)}]
                              </Text>
                            )}
                          </VStack>
                        </Table.Cell>
                        <Table.Cell textAlign="right">
                          <VStack align="end" gap={0}>
                            <Text>{row.improvability.value.toFixed(3)}</Text>
                            {row.improvability.ci && (
                              <Text fontSize="xs" color="fg.muted">
                                [{row.improvability.ci[0].toFixed(3)}, {row.improvability.ci[1].toFixed(3)}]
                              </Text>
                            )}
                          </VStack>
                        </Table.Cell>
                        <Table.Cell textAlign="right" display={{ base: "none", md: "table-cell" }}>
                          <Text>{row.rank.value.toFixed(2)}</Text>
                        </Table.Cell>
                        <Table.Cell textAlign="right" display={{ base: "none", md: "table-cell" }}>
                          <Text>{row.mrr.value.toFixed(3)}</Text>
                        </Table.Cell>
                      </Table.Row>
                    ))}
                  </Table.Body>
                </Table.Root>
              )}
            </GridItem>

          </Grid>
        </Container>
      </Box>
    </Box >
  )
}
