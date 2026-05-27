import { FormEvent, useState } from 'react';
import { Bitcoin, LockKeyhole } from 'lucide-react';
import { Button } from './Button';
import { Badge } from './Badge';

type LoginScreenProps = {
  onSaveToken: (token: string) => void;
};

export function LoginScreen({ onSaveToken }: LoginScreenProps) {
  const [token, setToken] = useState('');

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = token.trim();
    if (trimmed) {
      onSaveToken(trimmed);
      setToken('');
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <div className="glass-card w-full max-w-md rounded-3xl p-7 shadow-glow">
        <div className="mb-8 flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-btc-500 text-black">
            <Bitcoin className="h-7 w-7" />
          </div>
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-btc-400">Private dashboard</p>
            <h1 className="text-2xl font-bold text-white">BTC ML Paper Trader</h1>
          </div>
        </div>

        <div className="mb-6 flex flex-wrap gap-2">
          <Badge tone="orange">Paper Trading Only</Badge>
          <Badge tone="blue">BTC/USD Only</Badge>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <label className="block">
            <span className="mb-2 flex items-center gap-2 text-sm font-medium text-zinc-300">
              <LockKeyhole className="h-4 w-4 text-btc-400" />
              Admin token
            </span>
            <input
              autoComplete="off"
              className="h-12 w-full rounded-2xl border border-white/10 bg-black/30 px-4 text-white outline-none ring-btc-400/40 transition placeholder:text-zinc-600 focus:border-btc-400/60 focus:ring-4"
              onChange={(event) => setToken(event.target.value)}
              placeholder="Enter X-Admin-Token"
              type="password"
              value={token}
            />
          </label>
          <Button className="w-full" disabled={!token.trim()} type="submit" variant="primary">
            Open Dashboard
          </Button>
        </form>

        <p className="mt-5 text-xs leading-5 text-zinc-500">
          The token is stored only in this browser tab session via sessionStorage and is never displayed after saving.
        </p>
      </div>
    </main>
  );
}
