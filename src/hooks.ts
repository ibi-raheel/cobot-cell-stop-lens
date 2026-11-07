import { useEffect, useRef, useState } from "react";

export function useReducedMotion(): boolean {
  const q = "(prefers-reduced-motion: reduce)";
  const [r, setR] = useState(() => typeof window !== "undefined" && window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const on = () => setR(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return r;
}

/** Width of an element, kept current with ResizeObserver. */
export function useWidth<T extends HTMLElement>(fallback = 640): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(el);
    setW(Math.round(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Tween a numeric vector toward `target`; interruptible (restarts from the current value). */
export function useTween(target: number[], ms: number, instant: boolean): number[] {
  const [cur, setCur] = useState(target);
  const curRef = useRef(target);
  const key = target.map((v) => v.toFixed(4)).join(",");
  useEffect(() => {
    if (instant) {
      curRef.current = target;
      setCur(target);
      return;
    }
    const from = curRef.current;
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const p = Math.min(1, (now - t0) / ms);
      const e = easeInOut(p);
      const v = from.map((a, i) => a + (target[i] - a) * e);
      curRef.current = v;
      setCur(v);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, ms, instant]);
  return cur;
}

export function useInView<T extends Element>(margin = "0px"): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [v, setV] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setV(e.isIntersecting), { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);
  return [ref, v];
}
