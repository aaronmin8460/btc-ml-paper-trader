import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

type BadgeTone = 'orange' | 'green' | 'blue' | 'yellow' | 'red' | 'gray' | 'neutral';

const toneClasses: Record<BadgeTone, string> = {
  orange: 'border-btc-400/30 bg-btc-500/15 text-btc-400',
  green: 'border-emerald-400/30 bg-emerald-500/15 text-emerald-300',
  blue: 'border-sky-400/30 bg-sky-500/15 text-sky-300',
  yellow: 'border-amber-400/30 bg-amber-500/15 text-amber-200',
  red: 'border-rose-400/30 bg-rose-500/15 text-rose-300',
  gray: 'border-white/10 bg-white/10 text-zinc-300',
  neutral: 'border-white/10 bg-white/10 text-zinc-200',
};

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: BadgeTone }) {
  return (
    <span className={cn('inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold', toneClasses[tone])}>
      {children}
    </span>
  );
}
