// Copy and derived counters. Measured numbers live in metrics.gen.ts.

export {
  ABLATION,
  ARCH,
  ILLUSTRATIVE,
  JOINT,
  METRICS,
  PROTOCOL,
} from './metrics.gen'

import { METRICS, PROTOCOL } from './metrics.gen'

export type Counter = { key: string; label: string; value: number; unit: string; note: string }
export const COUNTERS: Counter[] = [
  {
    key: 'success',
    label: 'Intervention success',
    value: METRICS.successPct,
    unit: '%',
    note: 'named on/off checks that behaved as claimed',
  },
  {
    key: 'task',
    label: 'Copy-puzzle accuracy',
    value: METRICS.taskPct,
    unit: '%',
    note: `copy puzzle on ${PROTOCOL.nEval} sequences`,
  },
  {
    key: 'random',
    label: 'Random heads',
    value: METRICS.randomPct,
    unit: '%',
    note: 'same ops, heads I did not name',
  },
  {
    key: 'chance',
    label: 'Chance',
    value: METRICS.chancePct,
    unit: '%',
    note: `1 / vocab (${PROTOCOL.vocab})`,
  },
]

export const DECISIONS = [
  {
    first: 'Treat 100% puzzle accuracy as the headline',
    built: 'Report how often named on/off checks behave as claimed, with random heads next to it',
  },
  {
    first: 'Keep the MLP and hope a single head pops out',
    built: 'Train attention-only so the copy cannot hide in a feed-forward layer',
  },
  {
    first: 'Name heads after looking at the score',
    built: 'Lock the smallest joint set that breaks the copy, then measure restore vs leftover heads',
  },
  {
    first: 'Hide the random column',
    built: 'Print named-head success next to random-head success and chance on the same sequences',
  },
] as const

export type Tool = { name: string; tag: string; plain: string; tech: string }
export const STACK: Tool[] = [
  {
    name: 'Synthetic copy task',
    tag: 'data',
    plain: 'Builds short sequences where the last token should copy the token after the first match.',
    tech: '[x, y, z, x] with gold y. Vocab 16, sequence length 4, frozen in const.py before scoring.',
  },
  {
    name: 'Tiny causal transformer',
    tag: 'model',
    plain: 'A one-layer attention-only net small enough that I can ablate every head.',
    tech: '4 heads, d_model 32, no MLP. Last-token cross-entropy on the copy.',
  },
  {
    name: 'Head ablation + patch',
    tag: 'intervene',
    plain: 'Zeros named heads to break the copy, then copies their clean activations onto a corrupted sequence to restore it.',
    tech: 'Corrupt changes position 1 (y). Patch writes clean z of the named set into the corrupt forward.',
  },
  {
    name: 'Locked suite',
    tag: 'eval',
    plain: 'Decides which ops should break, hold, restore, or fail before the percentage is counted.',
    tech: 'Smallest joint set whose ablation drops task accuracy by at least 20 points. Leftover head is the negative control. Random heads are the debug column.',
  },
  {
    name: 'Vite + React + Tailwind',
    tag: 'page',
    plain: 'Builds this walkthrough from a single metrics file.',
    tech: 'No live API. Charts are SVG that read metrics.gen.ts. motion rolls the bento counters and respects reduced motion.',
  },
  {
    name: 'motion',
    tag: 'motion',
    plain: 'Animates section numbers when they enter the viewport.',
    tech: 'Counter tween via motion/react animate(); no transform travel when prefers-reduced-motion is set.',
  },
]

export const NEXT = [
  'Try modular addition (mod 17) on the same intervention suite once copy is stable.',
  'Path-patch the copy from the named set into a two-layer net that has an MLP again.',
  'Count how often a leftover head starts carrying the copy after I retrain with a different seed.',
]

export const SECTORS = [
  {
    name: 'Safety interpretability',
    job: 'Check whether a claimed circuit actually restores or breaks behavior, not only whether the net scores well.',
  },
  {
    name: 'Education',
    job: 'Show induction-style copy with a head map a student can ablate by hand.',
  },
  {
    name: 'Debugging trained policies',
    job: 'Point at the heads that implement a known subroutine before editing them.',
  },
]
