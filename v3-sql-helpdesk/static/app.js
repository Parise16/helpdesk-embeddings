const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || "Erro na requisição.");
  }

  return payload;
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showPage(pageId) {
  $$(".page").forEach((page) => page.classList.toggle("active", page.id === pageId));
  $$(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.page === pageId));

  if (pageId === "dashboard") loadDashboard();
  if (pageId === "open-tickets") loadOpenTickets();
  if (pageId === "reviews") loadReviews();
  if (pageId === "team") loadTeam();
}

async function loadDashboard() {
  const data = await api("/api/dashboard");
  const metrics = [
    ["Chamados abertos", data.open_tickets],
    ["Concluídos", data.resolved_tickets],
    ["Classificações", data.predictions],
    ["Qwen utilizado", `${data.qwen_rate}%`],
    ["Sem revisão humana", data.needs_review],
  ];

  $("#metrics").innerHTML = metrics
    .map(([label, value]) => `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

async function loadOpenTickets() {
  const container = $("#open-ticket-list");

  try {
    const data = await api("/api/tickets/open");

    if (!data.tickets.length) {
      container.innerHTML = `<div class="panel empty-state"><div><h2>Nenhum chamado aberto</h2><p>A fila está vazia.</p></div></div>`;
      return;
    }

    container.innerHTML = data.tickets
      .map(
        (ticket) => `
          <article class="ticket-card">
            <div>
              <div class="ticket-meta">
                <span class="badge">#${ticket.id}</span>
                <span>${escapeHtml(ticket.priority)}</span>
                <span>${escapeHtml(ticket.category_name || "Sem categoria")}</span>
              </div>
              <h3>${escapeHtml(ticket.title)}</h3>
              <div class="ticket-meta">
                <span>${escapeHtml(ticket.department_name || "Sem departamento")}</span>
                <span>Responsável: ${escapeHtml(ticket.employee_name || "—")}</span>
                <span>${escapeHtml(ticket.extracted_location || "Local não informado")}</span>
              </div>
              <p>${escapeHtml(ticket.problem_summary || ticket.description)}</p>
            </div>
            <button class="resolve-button" data-resolve="${ticket.id}">Marcar como concluído</button>
          </article>
        `,
      )
      .join("");

    $$("[data-resolve]").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await api(`/api/tickets/${button.dataset.resolve}/resolve`, { method: "PATCH" });
          await Promise.all([loadOpenTickets(), loadDashboard(), loadTeam()]);
        } catch (error) {
          button.disabled = false;
          alert(error.message);
        }
      });
    });
  } catch (error) {
    container.innerHTML = `<div class="panel message-box error-box">${escapeHtml(error.message)}</div>`;
  }
}

async function loadReviews() {
  const stats = $("#review-stats");
  const container = $("#review-container");

  try {
    const [reviewData, categoryData] = await Promise.all([
      api("/api/reviews/pending?limit=1"),
      api("/api/categories"),
    ]);

    stats.innerHTML = `
      <div class="metric-card"><span>Pendentes</span><strong>${reviewData.pending_count}</strong></div>
      <div class="metric-card"><span>Revisadas</span><strong>${reviewData.reviewed_count}</strong></div>
    `;

    if (!reviewData.items.length) {
      container.innerHTML = `
        <div class="panel empty-state">
          <div>
            <h2>Revisão concluída</h2>
            <p>Não existem previsões aguardando revisão humana.</p>
          </div>
        </div>
      `;
      return;
    }

    const item = reviewData.items[0];
    const categories = categoryData.categories
      .map(
        (category) => `
          <option value="${category.id}" ${category.id === item.predicted_category_id ? "disabled" : ""}>
            ${escapeHtml(category.name)}
          </option>
        `,
      )
      .join("");

    const nlp = item.nlp || {};
    const technologies = (nlp.tecnologias || []).join(", ") || "—";
    const devices = (nlp.dispositivos || []).join(", ") || "—";
    const systems = (nlp.sistemas || []).join(", ") || "—";

    container.innerHTML = `
      <article class="panel review-card">
        <div class="review-heading">
          <div>
            <div class="ticket-meta">
              <span class="badge">Ticket #${item.ticket_id}</span>
              <span>Previsão #${item.prediction_id}</span>
              <span>${item.qwen_used ? "Qwen utilizado" : "Sem Qwen"}</span>
            </div>
            <h2>${escapeHtml(item.title)}</h2>
            <p class="review-description">${escapeHtml(item.description)}</p>
          </div>
          <div class="predicted-category">
            <span>Categoria prevista</span>
            <strong>${escapeHtml(item.predicted_category)}</strong>
          </div>
        </div>

        <div class="detail-grid review-details">
          <div class="detail-item"><span>Embedding</span>${percent(item.semantic_score)}</div>
          <div class="detail-item"><span>NLP</span>${percent(item.nlp_score)}</div>
          <div class="detail-item"><span>Híbrido</span>${percent(item.hybrid_score)}</div>
          <div class="detail-item"><span>Margem</span>${percent(item.margin)}</div>
          <div class="detail-item"><span>Origem</span>${escapeHtml(item.decision_source)}</div>
          <div class="detail-item"><span>Local</span>${escapeHtml(item.extracted_location || "Não identificado")}</div>
        </div>

        <div class="review-context">
          <div><span>Resumo do problema</span><p>${escapeHtml(item.problem_summary)}</p></div>
          <div><span>Tecnologia</span><p>${escapeHtml(technologies)}</p></div>
          <div><span>Dispositivo</span><p>${escapeHtml(devices)}</p></div>
          <div><span>Sistema</span><p>${escapeHtml(systems)}</p></div>
        </div>

        <div class="review-actions">
          <button class="review-correct-button" id="review-correct">✓ Classificação correta</button>
          <button class="review-wrong-button" id="review-wrong">✕ Classificação incorreta</button>
        </div>

        <form class="review-correction-form hidden" id="review-correction-form">
          <label>
            Categoria correta
            <select id="review-category" required>
              <option value="">Selecione...</option>
              ${categories}
            </select>
          </label>
          <label>
            Observação
            <textarea id="review-notes" rows="3" maxlength="1000" placeholder="Opcional: explique por que a classificação estava errada."></textarea>
          </label>
          <div class="review-form-actions">
            <button class="secondary-button" type="button" id="cancel-review-correction">Cancelar</button>
            <button class="primary-button" type="submit">Salvar correção</button>
          </div>
        </form>
      </article>
    `;

    $("#review-correct").addEventListener("click", async () => {
      await submitReview(item.prediction_id, {
        is_correct: true,
        corrected_category_id: null,
        notes: "Revisado manualmente como correto.",
      });
    });

    $("#review-wrong").addEventListener("click", () => {
      $("#review-correction-form").classList.remove("hidden");
      $("#review-wrong").disabled = true;
    });

    $("#cancel-review-correction").addEventListener("click", () => {
      $("#review-correction-form").classList.add("hidden");
      $("#review-wrong").disabled = false;
    });

    $("#review-correction-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const categoryId = Number($("#review-category").value);
      if (!categoryId) return;

      await submitReview(item.prediction_id, {
        is_correct: false,
        corrected_category_id: categoryId,
        notes: $("#review-notes").value || null,
      });
    });
  } catch (error) {
    stats.innerHTML = "";
    container.innerHTML = `<div class="panel message-box error-box">${escapeHtml(error.message)}</div>`;
  }
}

async function submitReview(predictionId, payload) {
  const buttons = $$("#review-container button");
  buttons.forEach((button) => { button.disabled = true; });

  try {
    await api(`/api/predictions/${predictionId}/feedback`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await Promise.all([loadReviews(), loadDashboard()]);
  } catch (error) {
    buttons.forEach((button) => { button.disabled = false; });
    alert(error.message);
  }
}

async function loadTeam() {
  const data = await api("/api/team");
  $("#team-list").innerHTML = data.team
    .map(
      (member) => `
        <article class="team-card">
          <h3>${escapeHtml(member.name)}</h3>
          <div class="team-meta">${escapeHtml(member.department_name)}</div>
          <div class="team-load">${member.open_tickets}</div>
          <div class="team-meta">chamados em aberto</div>
        </article>
      `,
    )
    .join("");
}

$("#ticket-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const form = event.currentTarget;
  const button = form.querySelector("button[type=submit]");
  const result = $("#ticket-result");
  button.disabled = true;
  button.textContent = "Analisando...";
  result.innerHTML = `<div class="empty-state"><div><h2>Processando</h2><p>NLP, embeddings, roteamento e possível fallback com Qwen.</p></div></div>`;

  try {
    const payload = {
      requester_name: $("#requester-name").value,
      requester_email: $("#requester-email").value || null,
      title: $("#ticket-title").value,
      description: $("#ticket-description").value,
    };

    const data = await api("/api/tickets", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    result.innerHTML = `
      <h2>Chamado #${data.ticket_id}</h2>
      <div class="message-box">${escapeHtml(data.user_message)}</div>

      <div class="detail-grid">
        <div class="detail-item"><span>Categoria</span>${escapeHtml(data.category)}</div>
        <div class="detail-item"><span>Departamento</span>${escapeHtml(data.department)}</div>
        <div class="detail-item"><span>Responsável</span>${escapeHtml(data.employee)}</div>
        <div class="detail-item"><span>Chamados à frente</span>${data.queue_ahead}</div>
        <div class="detail-item"><span>Local</span>${escapeHtml(data.location || "Não identificado")}</div>
        <div class="detail-item"><span>Prioridade</span>${escapeHtml(data.priority)}</div>
        <div class="detail-item"><span>Qwen utilizado</span>${data.qwen_used ? "Sim" : "Não"}</div>
        <div class="detail-item"><span>Origem</span>${escapeHtml(data.decision_source)}</div>
      </div>

      <div class="candidates">
        <h2>Top categorias</h2>
        ${data.candidates
          .map(
            (candidate) => `
              <div class="candidate">
                <span>${escapeHtml(candidate.category_name)}</span>
                <strong>${percent(candidate.hybrid_score)}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
    `;

    form.reset();
    await Promise.all([loadDashboard(), loadOpenTickets(), loadTeam()]);
  } catch (error) {
    result.innerHTML = `<div class="message-box error-box">${escapeHtml(error.message)}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Criar e analisar chamado";
  }
});

$$(".nav-button").forEach((button) => {
  button.addEventListener("click", () => showPage(button.dataset.page));
});

$("#refresh-dashboard").addEventListener("click", loadDashboard);
$("#refresh-tickets").addEventListener("click", loadOpenTickets);
$("#refresh-reviews").addEventListener("click", loadReviews);
$("#refresh-team").addEventListener("click", loadTeam);

loadDashboard();
