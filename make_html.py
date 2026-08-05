import os

os.makedirs("static", exist_ok=True)

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Recruitment & Hiring Suite</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" />
  <style>
    :root { --ats-primary: #4f46e5; --ats-primary-dark: #4338ca; --ats-bg: #f8fafc; --ats-card: #ffffff; --ats-border: #e2e8f0; }
    body { background: var(--ats-bg); min-height: 100vh; }
    .navbar-brand { font-weight: 700; letter-spacing: -0.02em; }
    .nav-tabs .nav-link { font-weight: 600; color: #64748b; border: none; border-bottom: 3px solid transparent; border-radius: 0; }
    .nav-tabs .nav-link.active { color: var(--ats-primary); border-bottom-color: var(--ats-primary); background: transparent; }
    .card { border: 1px solid var(--ats-border); border-radius: 0.75rem; background: var(--ats-card); }
    .btn-primary { background: var(--ats-primary); border-color: var(--ats-primary); }
    .btn-primary:hover { background: var(--ats-primary-dark); border-color: var(--ats-primary-dark); }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark shadow-sm">
    <div class="container">
      <a class="navbar-brand d-flex align-items-center gap-2" href="#"><i class="bi bi-robot text-indigo-400"></i> AI Recruitment & Hiring Suite</a>
      <div class="text-white-50 small">
        <a href="/docs" target="_blank" class="text-white text-decoration-none me-3"><i class="bi bi-file-text"></i> API Docs</a>
        <a href="/health" target="_blank" class="text-white text-decoration-none"><i class="bi bi-heart-pulse"></i> Health</a>
      </div>
    </div>
  </nav>

  <div class="container py-4">
    <ul class="nav nav-tabs mb-4" id="atsTabs" role="tablist">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" id="jobs-tab" data-bs-toggle="tab" data-bs-target="#jobs-pane" type="button" role="tab"><i class="bi bi-briefcase me-1"></i> Job Generator & Questions</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="rank-tab" data-bs-toggle="tab" data-bs-target="#rank-pane" type="button" role="tab"><i class="bi bi-people me-1"></i> Resume Ranking</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="chat-tab" data-bs-toggle="tab" data-bs-target="#chat-pane" type="button" role="tab"><i class="bi bi-chat-dots me-1"></i> AI Assistant</button>
      </li>
    </ul>

    <div class="tab-content" id="atsTabsContent">
      <div class="tab-pane fade show active" id="jobs-pane" role="tabpanel" tabindex="0">
        <div class="row g-4">
          <div class="col-lg-6">
            <div class="card shadow-sm h-100">
              <div class="card-header bg-white py-3"><h5 class="mb-0"><i class="bi bi-magic text-primary me-2"></i>Generate Job Description</h5></div>
              <div class="card-body">
                <form id="jobGenerateForm">
                  <div class="mb-3">
                    <label for="rawInput" class="form-label">Job Prompt / Requirements</label>
                    <textarea class="form-control" id="rawInput" rows="4" placeholder="e.g. Senior Python FastAPI Developer with 5+ years experience, Docker, PostgreSQL, and async Python." required></textarea>
                  </div>
                  <button type="submit" class="btn btn-primary w-100"><i class="bi bi-cpu me-1"></i> Generate & Index Job via AI</button>
                </form>
                <div id="jobResult" class="mt-3"></div>
              </div>
            </div>
          </div>

          <div class="col-lg-6">
            <div class="card shadow-sm h-100">
              <div class="card-header bg-white py-3"><h5 class="mb-0"><i class="bi bi-question-circle text-primary me-2"></i>Generate Interview Questions</h5></div>
              <div class="card-body">
                <form id="questionForm">
                  <div class="mb-3">
                    <label for="questionJobId" class="form-label">Job ID (UUID)</label>
                    <input type="text" class="form-control" id="questionJobId" placeholder="Paste generated Job UUID here" required />
                  </div>
                  <button type="submit" class="btn btn-outline-primary w-100"><i class="bi bi-list-check me-1"></i> Generate Question Bank</button>
                </form>
                <div id="questionResult" class="mt-3"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
''')
print("Part 1 written")

with open("static/index.html", "a", encoding="utf-8") as f:
    f.write('''
      <div class="tab-pane fade" id="rank-pane" role="tabpanel" tabindex="0">
        <div class="card shadow-sm">
          <div class="card-header bg-white py-3"><h5 class="mb-0"><i class="bi bi-bar-chart-steps text-primary me-2"></i>Hybrid Resume Ranking</h5></div>
          <div class="card-body">
            <form id="rankForm">
              <div class="mb-3">
                <label for="rankJobId" class="form-label">Job ID (UUID)</label>
                <input type="text" class="form-control" id="rankJobId" placeholder="Paste Job UUID" required />
              </div>
              <div class="mb-3">
                <label for="candidatesJson" class="form-label">Candidates JSON (List of resumes)</label>
                <textarea class="form-control font-monospace" id="candidatesJson" rows="8" placeholder='[
  {
    "resume_id": "res-1",
    "student_id": "std-101",
    "student_name": "Jane Doe",
    "raw_text": "Experienced Python FastAPI engineer with 5 years of backend development, PostgreSQL, and Docker."
  }
]' required></textarea>
              </div>
              <button type="submit" class="btn btn-primary"><i class="bi bi-trophy me-1"></i> Rank Candidates</button>
            </form>
            <div id="rankResult" class="mt-4"></div>
          </div>
        </div>
      </div>

      <div class="tab-pane fade" id="chat-pane" role="tabpanel" tabindex="0">
        <div class="card shadow-sm">
          <div class="card-header bg-white py-3"><h5 class="mb-0"><i class="bi bi-chat-heart text-primary me-2"></i>AI Assistant</h5></div>
          <div class="card-body">
            <div id="chatHistory" class="border rounded p-3 mb-3 bg-light" style="height: 350px; overflow-y: auto;">
              <div class="text-muted text-center my-auto">Ask me anything about the recruitment suite, jobs, or candidates!</div>
            </div>
            <form id="chatForm" class="d-flex gap-2">
              <input type="text" class="form-control" id="chatInput" placeholder="Type a message..." required />
              <button type="submit" class="btn btn-primary px-4">Send</button>
            </form>
          </div>
        </div>
      </div>
    </div>
  </div>
''')


with open("static/index.html", "a", encoding="utf-8") as f:
    f.write('''    document.getElementById("rankForm").addEventListener("submit", function (e) {
      e.preventDefault();
      var jobId = document.getElementById("rankJobId").value.trim();
      var candidatesRaw = document.getElementById("candidatesJson").value.trim();
      var resultDiv = document.getElementById("rankResult");
      resultDiv.innerHTML = '<div class="text-muted"><span class="spinner-border spinner-border-sm me-2"></span>Ranking candidates...</div>';

      var candidatesObj;
      try {
        candidatesObj = JSON.parse(candidatesRaw);
      } catch (err) {
        resultDiv.innerHTML = '<div class="alert alert-danger">Invalid JSON format in candidates input.</div>';
        return;
      }

      fetch("/api/v1/jobs/" + jobId + "/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidates: candidatesObj })
      })
      .then(function(resp) {
        if (!resp.ok) throw new Error("Ranking failed");
        return resp.json();
      })
      .then(function(data) {
        var html = '<h5 class="mb-3">Ranking Leaderboard</h5><div class="list-group">';
        (data.candidates || []).forEach(function(c, idx) {
          var sb = c.score_breakdown || {};
          html +=
            '<div class="list-group-item">' +
              '<div class="d-flex justify-content-between align-items-center">' +
                '<div><strong>#' + (idx + 1) + ' ' + c.student_name + '</strong> <small class="text-muted">(' + c.student_id + ')</small></div>' +
                '<div class="badge bg-primary fs-6">Score: ' + (sb.weighted_total || 0).toFixed(2) + '</div>' +
              '</div>' +
              '<div class="small text-muted mt-1">TF-IDF: ' + (sb.tfidf_score || 0).toFixed(1) + ' | Keywords: ' + (sb.keyword_score || 0).toFixed(1) + ' | Vector: ' + (sb.vector_score || 0).toFixed(1) + '</div>' +
            '</div>';
        });
        html += '</div>';
        resultDiv.innerHTML = html;
      })
      .catch(function(err) {
        resultDiv.innerHTML = '<div class="alert alert-danger">' + err.message + '</div>';
      });
    });

    document.getElementById("chatForm").addEventListener("submit", function (e) {
      e.preventDefault();
      var inputEl = document.getElementById("chatInput");
      var msg = inputEl.value.trim();
      if (!msg) return;
      inputEl.value = "";

      var historyDiv = document.getElementById("chatHistory");
      historyDiv.innerHTML += '<div class="mb-2 text-end"><strong>You:</strong> ' + msg + '</div>';
      historyDiv.scrollTop = historyDiv.scrollHeight;

      fetch("/api/v1/chatbot/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      })
      .then(function(resp) {
        if (!resp.ok) throw new Error("Chat failed");
        return resp.json();
      })
      .then(function(data) {
        historyDiv.innerHTML += '<div class="mb-2 text-start text-primary"><strong>AI:</strong> ' + (data.reply || data.response || "No reply") + '</div>';
        historyDiv.scrollTop = historyDiv.scrollHeight;
      })
      .catch(function(err) {
        historyDiv.innerHTML += '<div class="mb-2 text-danger"><strong>Error:</strong> Failed to reach AI assistant.</div>';
      });
    });
  </script>
</body>
</html>
''')
print("static/index.html generated successfully.")


