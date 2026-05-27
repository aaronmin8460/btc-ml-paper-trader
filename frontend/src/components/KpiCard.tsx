import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

type KpiCardProps = {
  label: string;
  value: string;
  icon: ReactNode;
  helper?: string;
  tone?: 'neutral' | 'green' | 'red' | 'orange' | 'blue';
};

const tones = {
  neutral: 'text-zinc-300',
  green: 'text-emerald-300',
  red: 'text-rose-300',
  orange: 'text-btc-400',
  blue: 'text-sky-300',
};

export function KpiCard({ label, value, icon, helper, tone = 'neutral' }: KpiCardProps) {
  return (
    <div className="glass-card rounded-2xl p-4">
      <div className="flex items-center justify-between gap-4">
        <p className="text-xs font-medium uppercase tracking-[0.16em] text-zinc-500">{label}</p>
        <div className={cn('rounded-xl bg-white/10 p-2', tones[tone])}>{icon}</div>
      </div>
      <p className={cn('mt-4 truncate text-2xl font-bold', tones[tone])}>{value}</p>
      {helper ? <p className="mt-1 truncate text-xs text-zinc-500">{helper}</p> : null}
    </div>
  );
}
