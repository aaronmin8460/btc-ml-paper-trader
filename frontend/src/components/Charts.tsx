import type { ReactElement } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { DashboardOrder, DashboardSignal, EquityPoint } from '../api/client';
import { formatCompactTime, formatUsd } from '../lib/formatters';
import { Card, CardHeader } from './Card';
import { EmptyState } from './EmptyState';

export function DashboardCharts({
  equityCurve,
  signals,
  orders,
}: {
  equityCurve: EquityPoint[];
  signals: DashboardSignal[];
  orders: DashboardOrder[];
}) {
  return (
    <section className="grid gap-4 xl:grid-cols-3">
      <EquityCurveChart data={equityCurve} />
      <SignalProbabilityChart signals={signals} />
      <OrderActivityChart orders={orders} />
    </section>
  );
}

function EquityCurveChart({ data }: { data: EquityPoint[] }) {
  const points = data
    .filter((point) => point.timestamp && point.cumulative_realized_pnl !== null)
    .map((point) => ({
      time: formatCompactTime(point.timestamp),
      pnl: point.cumulative_realized_pnl,
      drawdown: point.drawdown,
    }));

  return (
    <Card className="min-h-[340px]">
      <CardHeader eyebrow="Realized PnL" title="Equity Curve" />
      {points.length === 0 ? (
        <EmptyState title="No realized trade PnL yet" message="The chart appears after trades with stored pnl values exist." />
      ) : (
        <ChartFrame>
          <LineChart data={points}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis dataKey="time" stroke="#71717a" tickLine={false} />
            <YAxis stroke="#71717a" tickFormatter={(value) => formatUsd(Number(value), { maximumFractionDigits: 0 })} tickLine={false} />
            <Tooltip content={<ChartTooltip />} />
            <Line dataKey="pnl" dot={false} name="Cumulative PnL" stroke="#f7931a" strokeWidth={2.5} type="monotone" />
            <Line dataKey="drawdown" dot={false} name="Drawdown" stroke="#fb7185" strokeWidth={1.8} type="monotone" />
          </LineChart>
        </ChartFrame>
      )}
    </Card>
  );
}

function SignalProbabilityChart({ signals }: { signals: DashboardSignal[] }) {
  const points = signals
    .filter((signal) => signal.created_at && signal.buy_probability !== null && signal.sell_probability !== null)
    .slice()
    .reverse()
    .map((signal) => ({
      time: formatCompactTime(signal.created_at),
      buy: signal.buy_probability,
      sell: signal.sell_probability,
    }));

  return (
    <Card className="min-h-[340px]">
      <CardHeader eyebrow="Model / rules" title="Signal Probabilities" />
      {points.length === 0 ? (
        <EmptyState title="No signal history yet" message="Probabilities appear after the backend records signals." />
      ) : (
        <ChartFrame>
          <LineChart data={points}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis dataKey="time" stroke="#71717a" tickLine={false} />
            <YAxis domain={[0, 1]} stroke="#71717a" tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tickLine={false} />
            <Tooltip content={<ChartTooltip percent />} />
            <Legend />
            <Line dataKey="buy" dot={false} name="Buy probability" stroke="#34d399" strokeWidth={2.2} type="monotone" />
            <Line dataKey="sell" dot={false} name="Sell probability" stroke="#fb7185" strokeWidth={2.2} type="monotone" />
          </LineChart>
        </ChartFrame>
      )}
    </Card>
  );
}

function OrderActivityChart({ orders }: { orders: DashboardOrder[] }) {
  const grouped = orders
    .filter((order) => order.created_at)
    .slice()
    .reverse()
    .reduce<Record<string, { time: string; buys: number; sells: number }>>((acc, order) => {
      const time = formatCompactTime(order.created_at);
      acc[time] ??= { time, buys: 0, sells: 0 };
      if (order.side.toLowerCase() === 'sell') acc[time].sells += 1;
      else acc[time].buys += 1;
      return acc;
    }, {});
  const points = Object.values(grouped).slice(-40);

  return (
    <Card className="min-h-[340px]">
      <CardHeader eyebrow="Execution" title="Order Activity" />
      {points.length === 0 ? (
        <EmptyState title="No paper orders yet" message="Order activity appears only after recorded paper order attempts." />
      ) : (
        <ChartFrame>
          <BarChart data={points}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis dataKey="time" stroke="#71717a" tickLine={false} />
            <YAxis allowDecimals={false} stroke="#71717a" tickLine={false} />
            <Tooltip content={<ChartTooltip />} />
            <Legend />
            <Bar dataKey="buys" fill="#34d399" name="Buy orders" radius={[6, 6, 0, 0]} />
            <Bar dataKey="sells" fill="#fb7185" name="Sell orders" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ChartFrame>
      )}
    </Card>
  );
}

function ChartFrame({ children }: { children: ReactElement }) {
  return (
    <div className="h-[250px] w-full">
      <ResponsiveContainer height="100%" width="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}

function ChartTooltip({ active, payload, label, percent }: { active?: boolean; payload?: Array<{ name?: string; value?: number; color?: string }>; label?: string; percent?: boolean }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/95 p-3 text-xs shadow-xl">
      <p className="mb-2 font-semibold text-white">{label}</p>
      {payload.map((item) => (
        <div key={`${item.name}-${item.value}`} className="flex min-w-36 items-center justify-between gap-4">
          <span style={{ color: item.color }}>{item.name}</span>
          <span className="font-semibold text-zinc-100">{formatTooltipValue(item.value, percent)}</span>
        </div>
      ))}
    </div>
  );
}

function formatTooltipValue(value: number | undefined, percent?: boolean): string {
  if (value === undefined || value === null) return '—';
  if (percent) return `${(value * 100).toFixed(1)}%`;
  return Number.isInteger(value) ? `${value}` : value.toFixed(4);
}
