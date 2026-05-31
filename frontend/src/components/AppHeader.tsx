"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";

export function AppHeader({ children }: { children?: React.ReactNode }) {
  const { user, signOut } = useAuth();
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 border-b border-border/60 bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3.5">
        <div className="flex items-center gap-1">
          <Link
            href="/"
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              pathname === "/"
                ? "bg-surface-2 text-foreground"
                : "text-muted hover:text-foreground"
            }`}
          >
            Portfolio
          </Link>
          <Link
            href="/screener"
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              pathname === "/screener"
                ? "bg-surface-2 text-foreground"
                : "text-muted hover:text-foreground"
            }`}
          >
            Screener
          </Link>
          {children}
        </div>
        <div className="flex items-center gap-4 text-xs text-muted">
          {user?.email && (
            <span className="hidden rounded-full border border-border bg-surface px-3 py-1 font-medium sm:inline">
              {user.email}
            </span>
          )}
          <button onClick={signOut} className="transition-colors hover:text-foreground">
            Sign out
          </button>
        </div>
      </div>
    </header>
  );
}
