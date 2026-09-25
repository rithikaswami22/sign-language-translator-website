const API_BASE = "";

const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");
const showRegisterLink = document.getElementById("show-register");
const messageEl = document.getElementById("auth-message");
const strengthEl = document.getElementById("password-strength");
const requirementsEl = document.getElementById("password-requirements");
const regPasswordInput = document.getElementById("reg-password");

if (showRegisterLink) {
  showRegisterLink.addEventListener("click", (e) => {
    e.preventDefault();
    registerForm.classList.toggle("hidden");
  });
}

if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    messageEl.textContent = "";

    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;

    try {
      const res = await fetch(`${API_BASE}/api/login`, {

        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        messageEl.textContent = data.error || "Login failed";
        messageEl.classList.add("error");
        return;
      }

      // Save user in localStorage for simple auth state
      try {
        localStorage.setItem("user", JSON.stringify(data));
      } catch (e) {
        console.warn("Could not save user to localStorage", e);
      }

      messageEl.textContent = "Logged in successfully!";
      messageEl.classList.remove("error");
      messageEl.classList.add("success");

      // Redirect to practice page after login
      const params = new URLSearchParams(window.location.search);
      const redirectTo = params.get("redirect") || "home.html";
      window.location.href = redirectTo;
    } catch (err) {
      messageEl.textContent = "Could not reach server. Is it running on port 4000?";
      messageEl.classList.add("error");
    }
  });
}

if (registerForm) {
  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    messageEl.textContent = "";

    const name = document.getElementById("reg-name").value.trim();
    const email = document.getElementById("reg-email").value.trim();
    const password = document.getElementById("reg-password").value;
    const issues = passwordIssues(password);
    if (issues.length) {
      messageEl.textContent = `Password must include ${issues.join(", ")}.`;
      messageEl.classList.add("error");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/api/register`, {

        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name, email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        messageEl.textContent = data.error || "Registration failed";
        messageEl.classList.add("error");
        return;
      }

      messageEl.textContent = "Account created! You can log in now.";
      messageEl.classList.remove("error");
      messageEl.classList.add("success");
      registerForm.reset();
      registerForm.classList.add("hidden");
    } catch (err) {
      messageEl.textContent = "Could not reach server. Is it running on port 4000?";
      messageEl.classList.add("error");
    }
  });
}

function passwordIssues(password) {
  const issues = [];
  if (!password || password.length < 8) issues.push("at least 8 characters");
  if (!/[a-z]/.test(password || "")) issues.push("a lowercase letter");
  if (!/[A-Z]/.test(password || "")) issues.push("an uppercase letter");
  if (!/[0-9]/.test(password || "")) issues.push("a number");
  if (!/[^\w\s]/.test(password || "")) issues.push("a special character");
  return issues;
}

function updatePasswordStrength() {
  if (!strengthEl || !requirementsEl || !regPasswordInput) return;
  const value = regPasswordInput.value || "";
  const issues = passwordIssues(value);
  const metCount = 5 - issues.length;

  strengthEl.classList.remove("weak", "ok", "strong");
  if (!value) {
    strengthEl.textContent = "Password strength:";
    requirementsEl.textContent = "";
    return;
  }

  if (metCount <= 2) {
    strengthEl.textContent = "Password strength: weak";
    strengthEl.classList.add("weak");
  } else if (metCount <= 4) {
    strengthEl.textContent = "Password strength: okay";
    strengthEl.classList.add("ok");
  } else {
    strengthEl.textContent = "Password strength: strong";
    strengthEl.classList.add("strong");
  }

  if (issues.length) {
    requirementsEl.textContent = `Needs: ${issues.join(", ")}.`;
  } else {
    requirementsEl.textContent = "All requirements met.";
  }
}

if (regPasswordInput) {
  regPasswordInput.addEventListener("input", updatePasswordStrength);
  updatePasswordStrength();
}

