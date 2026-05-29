import { Activity, BadgeDollarSign, BarChart3, Bitcoin, CircleDollarSign, Landmark, LineChart, ListOrdered, Percent, ShieldCheck, Signal, Wallet } from 'lucide-react';
import type { DashboardSummary } from '../api/client';
import { formatNumber, formatPercent, formatUsd } from '../lib/formatters';
import { KpiCard } from './KpiCard';

export function KpiGrid({ summary }: { summary: DashboardSummary }) {
  const position = summary.current_position;
  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <KpiCard icon={<Landmark className="h-4 w-4" />} label="Paper equity" value={formatUsd(summary.alpaca_account?.equity)} tone="orange" />
      <KpiCard icon={<Wallet className="h-4 w-4" />} label="Buying power" value={formatUsd(summary.alpaca_account?.buying_power)} />
      <KpiCard icon={<Bitcoin className="h-4 w-4" />} label="BTC quantity" value={formatNumber(position.qty, 8)} helper="Current paper position" tone="blue" />
      <KpiCard icon={<CircleDollarSign className="h-4 w-4" />} label="Market value" value={formatUsd(position.market_value)} />
      <KpiCard icon={<BadgeDollarSign className="h-4 w-4" />} label="Avg entry" value={formatUsd(position.avg_entry_price)} />
      <KpiCard icon={<LineChart className="h-4 w-4" />} label="Latest BTC price" value={formatUsd(summary.latest_btc_price)} tone="orange" />
      <KpiCard icon={<Activity className="h-4 w-4" />} label="Realized PnL" value={formatUsd(summary.total_realized_pnl)} tone={(summary.total_realized_pnl ?? 0) >= 0 ? 'green' : 'red'} />
      <KpiCard icon={<Activity className="h-4 w-4" />} label="Unrealized PnL" value={formatUsd(summary.unrealized_pnl)} tone={(summary.unrealized_pnl ?? 0) >= 0 ? 'green' : 'red'} />
      <KpiCard
        icon={<ShieldCheck className="h-4 w-4" />}
        label="Profit guard"
        value={summary.profit_guard_enabled ? (summary.profit_guard_exit_allowed ? 'Exit allowed' : 'Waiting') : 'Off'}
        helper={`PnL ${formatPercent(summary.current_unrealized_pnl_pct)} · min ${formatPercent(summary.min_net_exit_profit_pct)}`}
        tone={summary.profit_guard_exit_allowed ? 'green' : 'blue'}
      />
      <KpiCard
        icon={<ShieldCheck className="h-4 w-4" />}
        label="Min exit"
        value={formatUsd(summary.minimum_profitable_exit_price)}
        helper={`Estimate ${formatUsd(summary.estimated_exit_price)}`}
        tone="blue"
      />
      <KpiCard icon={<Percent className="h-4 w-4" />} label="Total return" value={formatPercent(summary.total_return_pct)} />
      <KpiCard icon={<Percent className="h-4 w-4" />} label="Win rate" value={formatPercent(summary.win_rate)} />
      <KpiCard icon={<ListOrdered className="h-4 w-4" />} label="Total orders" value={formatNumber(summary.total_orders, 0)} helper={`${summary.total_buy_orders} buys / ${summary.total_sell_orders} sells`} />
      <KpiCard icon={<BarChart3 className="h-4 w-4" />} label="Total trades" value={formatNumber(summary.total_trades, 0)} helper={`Latest: ${summary.latest_signal?.action?.toUpperCase() ?? '—'}`} />
      <KpiCard icon={<Signal className="h-4 w-4" />} label="Latest signal" value={summary.latest_signal?.action?.toUpperCase() ?? '—'} helper={summary.latest_signal?.reason ?? undefined} tone="blue" />
    </section>
  );
}
