import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '../lib/cn';

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

const variants: Record<ButtonVariant, string> = {
  primary: 'border-btc-400/40 bg-btc-500 text-black hover:bg-btc-400',
  secondary: 'border-white/10 bg-white/10 text-white hover:bg-white/15',
  danger: 'border-rose-400/30 bg-rose-500/15 text-rose-100 hover:bg-rose-500/25',
  ghost: 'border-transparent bg-transparent text-zinc-300 hover:bg-white/10 hover:text-white',
};

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: ButtonVariant;
  isLoading?: boolean;
};

export function Button({ children, className, variant = 'secondary', isLoading, disabled, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex h-10 items-center justify-center gap-2 rounded-xl border px-4 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50',
        variants[variant],
        className,
      )}
      disabled={disabled || isLoading}
      type="button"
      {...props}
    >
      {isLoading ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" /> : null}
      {children}
    </button>
  );
}
