import { Gauge, ShieldAlert, WalletCards } from 'lucide-react';
import type { ReactNode } from 'react';
import type { DashboardSummary } from '../api/client';
import { cn } from '../lib/cn';
import { formatNumber, formatPercent, formatUsd } from '../lib/formatters';
import { Card, CardHeader } from './Card';

export function RiskBudgetPanel({ summary }: { summary: DashboardSummary }) {
  const calls = summary.alpaca_calls_last_minute ?? 0;
  const remaining = summary.alpaca_budget_remaining;
  const totalBudget = remaining === null || remaining === undefined ? null : calls + remaining;
  const usedPct = totalBudget && totalBudget > 0 ? Math.min(1, calls / totalBudget) : null;
  const modelStatus =
    summary.active_model_valid
      ? 'Active accepted'
      : `Active ${summary.active_model_status}: ${summary.active_model_invalid_reason ?? summary.latest_model_rejected_reason ?? 'unknown'}`;

  return (
    <Card>
      <CardHeader eyebrow="Risk controls" title="API & Account Guard" />
      <div className="grid gap-4 lg:grid-cols-3">
        <GuardBlock
          icon={<Gauge className="h-4 w-4" />}
          label="Alpaca API budget"
          value={`${formatNumber(calls, 0)} calls/min`}
          helper={`${summary.api_budget_status ?? 'unknown'} · ${remaining === null || remaining === undefined ? '—' : `${remaining} left`}`}
          tone={summary.api_budget_status === 'hard_stop' ? 'red' : summary.api_budget_status === 'soft_limit' ? 'amber' : 'green'}
        >
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10">
            <div
              className={cn(
                'h-full rounded-full',
                summary.api_budget_status === 'hard_stop'
                  ? 'bg-rose-400'
                  : summary.api_budget_status === 'soft_limit'
                    ? 'bg-amber-300'
                    : 'bg-emerald-400',
              )}
              style={{ width: `${Math.round((usedPct ?? 0) * 100)}%` }}
            />
          </div>
        </GuardBlock>

        <GuardBlock
          icon={<WalletCards className="h-4 w-4" />}
          label="Paper account"
          value={formatUsd(summary.account_equity)}
          helper={`Cash ${formatUsd(summary.cash)} · BP ${formatUsd(summary.buying_power)}`}
          tone={(summary.account_daily_change_usd ?? 0) < 0 ? 'amber' : 'green'}
        />

        <GuardBlock
          icon={<ShieldAlert className="h-4 w-4" />}
          label="Account risk / model"
          value={formatPercent(summary.account_drawdown_pct)}
          helper={`Daily ${formatUsd(summary.account_daily_change_usd)} (${formatPercent(summary.account_daily_change_pct)}) · ${modelStatus}`}
          tone={(summary.account_drawdown_pct ?? 0) > 0.02 || !summary.active_model_valid ? 'amber' : 'blue'}
        />
      </div>
    </Card>
  );
}

function GuardBlock({
  children,
  helper,
  icon,
  label,
  tone,
  value,
}: {
  children?: ReactNode;
  helper: string;
  icon: ReactNode;
  label: string;
  tone: 'green' | 'amber' | 'red' | 'blue';
  value: string;
}) {
  return (
    <div className={cn('rounded-xl border bg-black/20 p-4', toneClass(tone))}>
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-zinc-400">
        {icon}
        {label}
      </div>
      <p className="mt-3 text-xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs text-zinc-400">{helper}</p>
      {children}
    </div>
  );
}

function toneClass(tone: 'green' | 'amber' | 'red' | 'blue'): string {
  if (tone === 'green') return 'border-emerald-400/20';
  if (tone === 'amber') return 'border-amber-400/25';
  if (tone === 'red') return 'border-rose-400/25';
  return 'border-sky-400/20';
}
