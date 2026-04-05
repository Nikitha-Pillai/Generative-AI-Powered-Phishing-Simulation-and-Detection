import { useEffect, useState } from "react";

const API_URL = "https://littleprophisher-phishing-backend.hf.space";

const cleanEmailContent = (text) => {
  if (!text) return "";

  if (text.trim().startsWith("<")) {
    const doc = new DOMParser().parseFromString(text, "text/html");
    doc.querySelectorAll("style, script, head").forEach(el => el.remove());
    text = doc.body.innerText || doc.body.textContent || "";
  }

  return text
    .replace(/&#847;|&zwnj;|&#8199;|&#65279;|&#173;|&#8203;|&#8204;|&#8205;/g, "")
    .replace(/[\u00AD\u200B\u200C\u200D\uFEFF\u034F\u2007]/g, "")
    .replace(/\s*<\s*$/gm, "")
    .replace(/[-]{5,}/g, "")
    .replace(/Forwarded message/g, "")
    .replace(/From:.*\n?/g, "")
    .replace(/Date:.*\n?/g, "")
    .replace(/To:.*\n?/g, "")
    .replace(/Subject:.*\n?/g, "")
    .replace(/^\s*\*\s*$/gm, "")
    .replace(/^\s*[•\-\*]\s*/gm, "")
    .replace(/^\s+$/gm, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
};

function Feedback() {
  const [emails, setEmails] = useState([]);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/api/low-confidence`)
      .then((res) => res.json())
      .then((data) => setEmails(data));
  }, []);

  const sendFeedback = (email_id, label) => {
    fetch(`${API_URL}/api/submit-feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email_id, user_label: label }),
    }).then(() => {
      alert("Feedback submitted!");
      setEmails(emails.filter((e) => e.email_id !== email_id));
    });
  };

  return (
    <div style={{
      maxWidth: "860px",
      margin: "0 auto",
      padding: "2rem 1.5rem",
      fontFamily: "var(--font-sans, system-ui, sans-serif)",
      fontSize: "15px",              // ← ADD: increase base font
      background: "var(--color-background-primary)",  // ← ADD: white background
      minHeight: "100vh",            // ← ADD: full height
    }}>
      {/* Header */}
      <div style={{ marginBottom: "2rem", textAlign: "center" }}>   {/* ← ADD textAlign center */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px", justifyContent: "center" }}>  {/* ← ADD justifyContent center */}
          <div style={{
            width: "32px", height: "32px", borderRadius: "8px",
            background: "var(--color-background-warning)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "16px",
          }}>⚠️</div>
          <h1 style={{
            fontSize: "26px",          // ← increase from 20px
            fontWeight: "500", margin: 0,
            color: "var(--color-text-primary)",
          }}>Emails needing review</h1>
        </div>
        <p style={{
          fontSize: "15px",            // ← increase from 14px
          color: "var(--color-text-secondary)",
          margin: "0",                 // ← remove left margin
        }}>
          {emails.length} emails flagged with low confidence — your feedback helps improve the model.
        </p>
      </div>

      {/* Empty state */}
      {emails.length === 0 ? (
        <div style={{
          textAlign: "center", padding: "3rem",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
          color: "var(--color-text-secondary)", fontSize: "14px",
        }}>
          No emails need review right now.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {emails.map((email) => {
            const isExpanded = expandedId === email.email_id;
            const isPhishing = email.prediction === "Phishing";
            const confidence = (email.confidence * 100).toFixed(2);

            return (
              <div key={email.email_id} style={{
                background: "var(--color-background-primary)",
                border: "0.5px solid var(--color-border-tertiary)",
                borderRadius: "var(--border-radius-lg)",
                overflow: "hidden",
              }}>
                {/* Card Header — clickable to expand */}
                <div
                  onClick={() => setExpandedId(isExpanded ? null : email.email_id)}
                  style={{
                    padding: "1rem 1.25rem",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    gap: "12px",
                  }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{
                      margin: "0 0 6px",
                      fontWeight: "500",
                      fontSize: "15px",
                      color: "var(--color-text-primary)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}>
                      {email.subject || "No Subject"}
                    </p>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                      {/* Prediction badge */}
                      <span style={{
                        fontSize: "12px",
                        fontWeight: "500",
                        padding: "2px 10px",
                        borderRadius: "var(--border-radius-md)",
                        background: isPhishing
                          ? "var(--color-background-danger)"
                          : "var(--color-background-success)",
                        color: isPhishing
                          ? "var(--color-text-danger)"
                          : "var(--color-text-success)",
                      }}>
                        {email.prediction}
                      </span>
                      {/* Confidence badge */}
                      <span style={{
                        fontSize: "12px",
                        color: "var(--color-text-secondary)",
                      }}>
                        {confidence}% confidence
                      </span>
                    </div>
                  </div>

                  {/* Chevron */}
                  <span style={{
                    fontSize: "12px",
                    color: "var(--color-text-tertiary)",
                    marginTop: "3px",
                    transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
                    transition: "transform 0.2s",
                    flexShrink: 0,
                  }}>▼</span>
                </div>

                {/* Expanded body */}
                {isExpanded && (
                  <div style={{
                    borderTop: "0.5px solid var(--color-border-tertiary)",
                    padding: "1rem 1.25rem",
                  }}>
                    {/* Email content */}
                    <p style={{
                      fontSize: "13px",
                      color: "var(--color-text-secondary)",
                      marginBottom: "6px",
                      fontWeight: "500",
                    }}>Email body</p>
                    <div style={{
                      background: "var(--color-background-secondary)",
                      borderRadius: "var(--border-radius-md)",
                      padding: "12px 14px",
                      marginBottom: "1.25rem",
                      fontSize: "13px",
                      color: "var(--color-text-primary)",
                      whiteSpace: "pre-line",
                      wordBreak: "break-word",
                      overflowWrap: "break-word",
                      lineHeight: "1.6",
                      maxHeight: "200px",
                      overflowY: "auto",
                    }}>
                      {cleanEmailContent(email.content)}
                    </div>

                    {/* Action buttons */}
                    <div style={{ display: "flex", gap: "8px" }}>
                      <button
                        onClick={() => sendFeedback(email.email_id, "Phishing")}
                        style={{
                          flex: 1,
                          padding: "8px 16px",
                          fontSize: "13px",
                          fontWeight: "500",
                          borderRadius: "var(--border-radius-md)",
                          border: "1.5px solid var(--color-border-danger)",
                          background: "#FCEBEB",
                          color: "#A32D2D",
                          cursor: "pointer",
                        }}
                      >
                        Mark as phishing
                      </button>
                      <button
                        onClick={() => sendFeedback(email.email_id, "Legitimate")}
                        style={{
                          flex: 1,
                          padding: "8px 16px",
                          fontSize: "13px",
                          fontWeight: "500",
                          borderRadius: "var(--border-radius-md)",
                          border: "1.5px solid var(--color-border-success)",
                          background: "#EAF3DE",
                          color: "#3B6D11",
                          cursor: "pointer",
                        }}
                      >
                        Mark as legitimate
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default Feedback;