import type { DashboardOrder } from '../api/client';
import { formatDateTime, formatNumber, formatUsd, statusColor } from '../lib/formatters';
import { cn } from '../lib/cn';
import { Card, CardHeader } from './Card';
import { EmptyState } from './EmptyState';

export function OrdersTable({ orders }: { orders: DashboardOrder[] }) {
  return (
    <Card>
      <CardHeader eyebrow="Execution" title="Recent Paper Orders" />
      {orders.length === 0 ? (
        <EmptyState title="No paper orders recorded" />
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-[0.14em] text-zinc-500">
              <tr className="border-b border-white/10">
                <th className="px-3 py-3">Time</th>
                <th className="px-3 py-3">Side</th>
                <th className="px-3 py-3">Status</th>
                <th className="px-3 py-3">Notional</th>
                <th className="px-3 py-3">Qty</th>
                <th className="px-3 py-3">Broker order id</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order, index) => (
                <tr className="border-b border-white/[0.06] last:border-0" key={`${order.broker_order_id ?? 'order'}-${index}`}>
                  <td className="whitespace-nowrap px-3 py-3 text-zinc-300">{formatDateTime(order.created_at)}</td>
                  <td className="px-3 py-3">
                    <span className={cn('rounded-full border px-2.5 py-1 text-xs font-semibold uppercase', statusColor(order.side))}>{order.side}</span>
                  </td>
                  <td className="px-3 py-3 text-zinc-300">{order.status ?? '—'}</td>
                  <td className="px-3 py-3 text-zinc-300">{formatUsd(order.notional)}</td>
                  <td className="px-3 py-3 text-zinc-300">{formatNumber(order.qty, 8)}</td>
                  <td className="max-w-[260px] truncate px-3 py-3 font-mono text-xs text-zinc-500">{order.broker_order_id ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
