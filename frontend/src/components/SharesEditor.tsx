"use client";

import { useEffect, useState } from "react";

/** Inline numeric editor that commits on blur or Enter. */
export function SharesEditor({
  value,
  onCommit,
}: {
  value: number;
  onCommit: (shares: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  function commit() {
    const n = Number(draft);
    if (!Number.isNaN(n) && n >= 0 && n !== value) {
      onCommit(n);
    } else {
      setDraft(String(value));
    }
  }

  return (
    <input
      type="number"
      min={0}
      step="any"
      value={draft}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
      className="nums w-20 rounded-md border border-transparent bg-transparent px-2 py-1 text-right text-sm hover:border-border focus:border-accent focus:bg-background focus:outline-none"
    />
  );
}
