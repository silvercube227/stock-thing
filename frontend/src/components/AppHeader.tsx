"use client";

import Link from "next/link";
import { useAuth } from "@/components/AuthProvider";

export function AppHeader({ children }: { children?: React.ReactNode }) {
  const { user, signOut } = useAuth();
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-sm font-semibold tracking-tight">
            Trends
          </Link>
          <Link href="/screener" className="text-xs text-muted hover:text-foreground">
            Screener
          </Link>
          {children}
        </div>
        <div className="flex items-center gap-3 text-xs text-muted">
          {user?.email && <span className="hidden sm:inline">{user.email}</span>}
          <button onClick={signOut} className="hover:text-foreground">
            Sign out
          </button>
        </div>
      </div>
    </header>
  );
}
