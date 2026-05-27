import { ArrowDownCircle, ArrowRightCircle, ArrowUpCircle } from 'lucide-react';
import type { DashboardMarket, DashboardSignal } from '../api/client';
import { cn } from '../lib/cn';
import { formatNumber, formatProbability, formatUsd, statusColor } from '../lib/formatters';
import { Card, CardHeader } from './Card';
import { EmptyState } from './EmptyState';

export function SignalPanel({ signal, market }: { signal: DashboardSignal | null; market: DashboardMarket | null }) {
  if (!signal) {
    return (
      <Card>
        <CardHeader eyebrow="Latest signal" title="Decision" />
        <EmptyState title="No signal yet" message="Run the bot once or start automation to record the first signal." />
      </Card>
    );
  }

  const action = signal.action.toUpperCase();
  const Icon = signal.action.toLowerCase() === 'buy' ? ArrowUpCircle : signal.action.toLowerCase() === 'sell' ? ArrowDownCircle : ArrowRightCircle;

  return (
    <Card>
      <CardHeader eyebrow="Latest signal" title="Decision" />
      <div className="flex items-center gap-3">
        <div className={cn('rounded-2xl border p-3', statusColor(signal.action))}>
          <Icon className="h-6 w-6" />
        </div>
        <div>
          <p className="text-3xl font-bold text-white">{action}</p>
          <p className="text-sm text-zinc-500">{signal.reason ?? '—'}</p>
        </div>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <SignalMetric label="Buy probability" value={formatProbability(signal.buy_probability)} />
        <SignalMetric label="Sell probability" value={formatProbability(signal.sell_probability)} />
        <SignalMetric label="Latest price" value={formatUsd(market?.latest_close)} />
        <SignalMetric label="Spread" value={market?.spread_bps === null || market?.spread_bps === undefined ? '—' : `${formatNumber(market.spread_bps, 2)} bps`} />
        <SignalMetric label="Quote imbalance" value={formatNumber(market?.quote_imbalance, 4)} />
        <SignalMetric label="Market timestamp" value={market?.latest_timestamp ? new Date(market.latest_timestamp).toLocaleTimeString() : '—'} />
      </div>
    </Card>
  );
}

function SignalMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-3">
      <p className="text-xs uppercase tracking-[0.16em] text-zinc-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-zinc-100">{value}</p>
    </div>
  );
}
