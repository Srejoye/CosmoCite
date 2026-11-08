export default function AttentionText({ words }: { words: { word: string; weight: number }[] }) {
  const max = Math.max(...words.map((w) => w.weight), 1e-9);
  return (
    <p style={{ lineHeight: 2 }}>
      {words.map((w, i) => (
        <span
          key={i}
          title={w.weight.toFixed(4)}
          style={{ background: `rgba(255,179,64,${(w.weight / max).toFixed(2)})`, color: w.weight / max > 0.6 ? "#05070f" : undefined, borderRadius: 4, padding: "1px 3px", marginRight: 2 }}
        >
          {w.word}
        </span>
      ))}
    </p>
  );
}