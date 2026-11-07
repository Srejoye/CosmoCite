import { Component, ReactNode } from "react";

/** Keeps a crash on one page from blanking the whole site. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main>
        <div className="card warn" role="alert">
          <p>This page hit a problem. Your other pages still work.</p>
          <button type="button" className="chip" style={{ marginTop: 12 }} onClick={() => location.reload()}>Reload</button>
        </div>
      </main>
    );
  }
}
