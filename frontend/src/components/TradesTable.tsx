import type { DashboardTrade } from '../api/client';
import { cn } from '../lib/cn';
import { formatDateTime, formatNumber, formatUsd, statusColor } from '../lib/formatters';
import { Card, CardHeader } from './Card';
import { EmptyState } from './EmptyState';

export function TradesTable({ trades }: { trades: DashboardTrade[] }) {
  return (
    <Card>
      <CardHeader eyebrow="PnL" title="Recent Trades" />
      {trades.length === 0 ? (
        <EmptyState title="No realized trades yet" message="Trade rows appear when the backend records realized paper-trade PnL." />
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-[0.14em] text-zinc-500">
              <tr className="border-b border-white/10">
                <th className="px-3 py-3">Time</th>
                <th className="px-3 py-3">Side</th>
                <th className="px-3 py-3">Qty</th>
                <th className="px-3 py-3">Price</th>
                <th className="px-3 py-3">PnL</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade, index) => (
                <tr className="border-b border-white/[0.06] last:border-0" key={`${trade.created_at ?? 'trade'}-${index}`}>
                  <td className="whitespace-nowrap px-3 py-3 text-zinc-300">{formatDateTime(trade.created_at)}</td>
                  <td className="px-3 py-3">
                    <span className={cn('rounded-full border px-2.5 py-1 text-xs font-semibold uppercase', statusColor(trade.side))}>{trade.side}</span>
                  </td>
                  <td className="px-3 py-3 text-zinc-300">{formatNumber(trade.qty, 8)}</td>
                  <td className="px-3 py-3 text-zinc-300">{formatUsd(trade.price)}</td>
                  <td className={cn('px-3 py-3 font-semibold', (trade.pnl ?? 0) >= 0 ? 'text-emerald-300' : 'text-rose-300')}>{formatUsd(trade.pnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
