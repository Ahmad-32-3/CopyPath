// Bars: task fit, named intervention success, random heads, chance.

import { METRICS } from '../../data'

const W = 340
const H = 240
const PAD = { l: 52, r: 16, t: 12, b: 56 }

const ROWS = [
  { key: 'task', label: 'task', value: METRICS.taskPct, fill: 'var(--chart-task)' },
  { key: 'named', label: 'named', value: METRICS.successPct, fill: 'var(--chart-named)' },
  { key: 'random', label: 'random', value: METRICS.randomPct, fill: 'var(--chart-random)' },
  { key: 'chance', label: 'chance', value: METRICS.chancePct, fill: 'var(--chart-chance)' },
] as const

export function ResultsBars() {
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const y = (v: number) => PAD.t + plotH - (v / 100) * plotH
  const bw = plotW / ROWS.length - 12

  return (
    <div className="chart-wrap">
      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Task accuracy ${METRICS.taskPct} percent, named intervention success ${METRICS.successPct} percent, random ${METRICS.randomPct} percent, chance ${METRICS.chancePct} percent.`}
      >
        {[0, 50, 85, 100].map((t) => (
          <g key={t}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--chart-grid)"
              strokeWidth="1"
              strokeDasharray={t === 85 ? '4 3' : undefined}
            />
            <text x={PAD.l - 6} y={y(t) + 3} textAnchor="end" fontSize="10" fill="var(--chart-label)" fontFamily="var(--font-mono)">
              {t}
            </text>
          </g>
        ))}
        {ROWS.map((row, i) => {
          const x = PAD.l + i * (plotW / ROWS.length) + 6
          const top = y(row.value)
          return (
            <g key={row.key}>
              <rect
                className="bar-grow"
                x={x}
                y={top}
                width={bw}
                height={Math.max(y(0) - top, 0.5)}
                fill={row.fill}
                style={{ animationDelay: `${i * 90}ms` }}
              />
              <text x={x + bw / 2} y={H - 28} textAnchor="middle" fontSize="11" fill="var(--chart-label)" fontFamily="var(--font-mono)">
                {row.label}
              </text>
              <text x={x + bw / 2} y={top - 6} textAnchor="middle" fontSize="10" fill="var(--fg-hi)" fontFamily="var(--font-mono)">
                {row.value}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
