import { useEffect, useState } from "react";
import { api, Stats } from "./api";

let cached: Promise<Stats> | null = null;

/** Live corpus size, fetched once per visit. Null until it loads or if the server is down. */
export function useStats() {
  const [stats, setStats] = useState<Stats | null>(null);
  useEffect(() => {
    let live = true;
    cached ??= api.stats();
    cached.then((s) => { if (live) setStats(s); }).catch(() => { cached = null; });
    return () => { live = false; };
  }, []);
  return stats;
}
