import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

type CardProps = {
  children: ReactNode;
  className?: string;
};

export function Card({ children, className }: CardProps) {
  return <section className={cn('glass-card rounded-2xl p-5', className)}>{children}</section>;
}

type CardHeaderProps = {
  title: string;
  eyebrow?: string;
  action?: ReactNode;
};

export function CardHeader({ title, eyebrow, action }: CardHeaderProps) {
  return (
    <div className="mb-4 flex items-start justify-between gap-3">
      <div>
        {eyebrow ? <p className="text-xs font-semibold uppercase tracking-[0.18em] text-btc-400/80">{eyebrow}</p> : null}
        <h2 className="mt-1 text-base font-semibold text-white">{title}</h2>
      </div>
      {action}
    </div>
  );
}
