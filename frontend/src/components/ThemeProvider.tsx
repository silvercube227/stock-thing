"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

type Theme = "dark" | "light";

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "dark",
  toggle: () => {},
});

export const THEME_STORAGE_KEY = "cardinal-theme";

/** Reads the theme the pre-hydration script already applied to <html> so the
 *  initial React state matches the DOM (no flash, no hydration mismatch). */
function readAppliedTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("dark");

  // Sync state to whatever the inline script already applied to <html>. This runs
  // AFTER the first client render on purpose: server + first client render both use
  // "dark" (matching, no hydration mismatch), then we reconcile to the real theme.
  // The one-time cascading render is the intended cost of SSR-safe theme sync.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTheme(readAppliedTheme());
  }, []);

  const toggle = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        /* private mode / storage disabled — theme still applies for this session */
      }
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
