interface UnifiedDiffViewProps {
  diff: string;
}

/**
 * Render a unified diff string (as returned by ``POST /api/diff``, M2-4)
 * with per-line +/- highlighting. Used by MetadataEditor and DocumentEditor
 * confirmation screens.
 */
export default function UnifiedDiffView({ diff }: UnifiedDiffViewProps) {
  const lines = diff.split("\n");
  return (
    <pre className="max-h-64 overflow-auto rounded bg-slate-900 p-2 font-mono text-xs">
      {lines.map((line, i) => {
        let className = "text-slate-400";
        if (line.startsWith("+++") || line.startsWith("---")) {
          className = "text-slate-500";
        } else if (line.startsWith("+")) {
          className = "text-green-400";
        } else if (line.startsWith("-")) {
          className = "text-red-400";
        } else if (line.startsWith("@@")) {
          className = "text-sky-400";
        }
        return (
          <div key={i} className={className}>
            {line.length > 0 ? line : " "}
          </div>
        );
      })}
    </pre>
  );
}
