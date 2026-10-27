/**
 * StockSense AI - Enterprise Authentication Controller
 * Handles client-side validation, password toggling, async API login,
 * demo credentials auto-fill, account creation, and session redirection.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Elements Cache ---
    const tabBtns = document.querySelectorAll(".auth-tab");
    const formPanes = document.querySelectorAll(".auth-form-pane");
    const authAlert = document.getElementById("authAlert");
    const authAlertIcon = document.getElementById("authAlertIcon");
    const authAlertText = document.getElementById("authAlertText");

    // Toast
    const toastNotification = document.getElementById("toastNotification");
    const toastMessage = document.getElementById("toastMessage");

    function showToast(msg) {
        if (!toastNotification || !toastMessage) return;
        toastMessage.textContent = msg;
        toastNotification.classList.add("show");
        setTimeout(() => toastNotification.classList.remove("show"), 3500);
    }

    function showAlert(type, message) {
        if (!authAlert) return;
        authAlert.className = `auth-alert auth-alert-${type} show`;
        if (type === "error") {
            authAlertIcon.className = "fa-solid fa-circle-exclamation";
        } else if (type === "success") {
            authAlertIcon.className = "fa-solid fa-circle-check";
        } else {
            authAlertIcon.className = "fa-solid fa-circle-info";
        }
        authAlertText.textContent = message;
    }

    function hideAlert() {
        if (!authAlert) return;
        authAlert.className = "auth-alert";
        authAlertText.textContent = "";
    }

    // --- Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetPaneId = btn.getAttribute("data-tab");
            switchTab(targetPaneId);
        });
    });

    function switchTab(paneId) {
        hideAlert();
        tabBtns.forEach(b => {
            const isActive = b.getAttribute("data-tab") === paneId;
            b.classList.toggle("active", isActive);
            b.setAttribute("aria-selected", isActive ? "true" : "false");
        });

        formPanes.forEach(p => {
            p.classList.toggle("active", p.id === paneId);
        });
    }

    // Link: Forgot Password from Sign In pane
    const linkForgotPassword = document.getElementById("linkForgotPassword");
    if (linkForgotPassword) {
        linkForgotPassword.addEventListener("click", () => {
            switchTab("pane-recover");
        });
    }

    // --- Password Visibility Toggles ---
    function setupPasswordToggle(btnId, inputId, iconId) {
        const btn = document.getElementById(btnId);
        const input = document.getElementById(inputId);
        const icon = document.getElementById(iconId);
        if (!btn || !input || !icon) return;

        btn.addEventListener("click", () => {
            const isPassword = input.type === "password";
            input.type = isPassword ? "text" : "password";
            icon.className = isPassword ? "fa-regular fa-eye-slash" : "fa-regular fa-eye";
            btn.setAttribute("aria-label", isPassword ? "Hide password" : "Show password");
        });
    }

    setupPasswordToggle("btnToggleSigninPassword", "signinPassword", "signinEyeIcon");
    setupPasswordToggle("btnToggleRegPassword", "regPassword", "regEyeIcon");

    // --- Demo Credentials Quick Fill ---
    const btnFillAnalyst = document.getElementById("btnFillAnalyst");
    const btnFillAdmin = document.getElementById("btnFillAdmin");
    const signinEmail = document.getElementById("signinEmail");
    const signinPassword = document.getElementById("signinPassword");

    if (btnFillAnalyst) {
        btnFillAnalyst.addEventListener("click", () => {
            switchTab("pane-signin");
            signinEmail.value = "demo@stocksense.ai";
            signinPassword.value = "StockSense2026!";
            signinEmail.classList.remove("is-invalid");
            signinPassword.classList.remove("is-invalid");
            document.getElementById("signinEmailError")?.classList.remove("show");
            document.getElementById("signinPasswordError")?.classList.remove("show");
            showAlert("info", "Lead Analyst demo credentials populated. Click 'Sign In' to proceed.");
        });
    }

    if (btnFillAdmin) {
        btnFillAdmin.addEventListener("click", () => {
            switchTab("pane-signin");
            signinEmail.value = "admin@stocksense.ai";
            signinPassword.value = "AdminStock#2026";
            signinEmail.classList.remove("is-invalid");
            signinPassword.classList.remove("is-invalid");
            document.getElementById("signinEmailError")?.classList.remove("show");
            document.getElementById("signinPasswordError")?.classList.remove("show");
            showAlert("info", "Supply Chain VP demo credentials populated. Click 'Sign In' to proceed.");
        });
    }

    // --- Check URL Params (e.g. ?logout=1 or ?next=...) ---
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("logout") === "1") {
        showToast("You have been signed out safely.");
    }
    if (urlParams.get("registered") === "1") {
        showAlert("success", "Account created successfully! You may now sign in.");
    }

    // --- Email Validator ---
    function isValidEmail(val) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(val).toLowerCase());
    }

    // =========================================================================
    // SIGN IN SUBMISSION HANDLER
    // =========================================================================
    const formSignIn = document.getElementById("pane-signin");
    const btnSignInSubmit = document.getElementById("btnSignInSubmit");
    const signinBtnIcon = document.getElementById("signinBtnIcon");
    const signinBtnText = document.getElementById("signinBtnText");

    const signinEmailError = document.getElementById("signinEmailError");
    const signinPasswordError = document.getElementById("signinPasswordError");

    if (formSignIn) {
        formSignIn.addEventListener("submit", async (e) => {
            e.preventDefault();
            hideAlert();

            const email = (signinEmail.value || "").trim();
            const password = (signinPassword.value || "");
            const rememberMe = document.getElementById("rememberMe")?.checked || false;

            let hasError = false;

            // Validate email
            if (!email || !isValidEmail(email)) {
                signinEmail.classList.add("is-invalid");
                signinEmailError.classList.add("show");
                hasError = true;
            } else {
                signinEmail.classList.remove("is-invalid");
                signinEmailError.classList.remove("show");
            }

            // Validate password
            if (!password) {
                signinPassword.classList.add("is-invalid");
                signinPasswordError.classList.add("show");
                hasError = true;
            } else {
                signinPassword.classList.remove("is-invalid");
                signinPasswordError.classList.remove("show");
            }

            if (hasError) return;

            // Set loading state
            btnSignInSubmit.disabled = true;
            signinEmail.disabled = true;
            signinPassword.disabled = true;
            signinBtnIcon.className = "fa-solid fa-circle-notch fa-spin";
            signinBtnText.textContent = "Authenticating...";

            try {
                const response = await fetch("/api/auth/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        email: email,
                        password: password,
                        remember_me: rememberMe
                    })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    showAlert("success", "Authentication verified! Redirecting to dashboard...");
                    signinBtnText.textContent = "Verified. Redirecting...";

                    // Get redirect target from URL query param or server response
                    const nextTarget = urlParams.get("next") || data.redirect || "/";
                    setTimeout(() => {
                        window.location.href = nextTarget;
                    }, 500);
                } else {
                    // Authentication error
                    const errMsg = data.error || "Authentication failed. Please verify your credentials.";
                    showAlert("error", errMsg);
                    signinPassword.value = "";
                    signinPassword.focus();

                    // Re-enable form controls
                    btnSignInSubmit.disabled = false;
                    signinEmail.disabled = false;
                    signinPassword.disabled = false;
                    signinBtnIcon.className = "fa-solid fa-arrow-right-to-bracket";
                    signinBtnText.textContent = "Sign In to Dashboard";
                }
            } catch (err) {
                showAlert("error", "Network or server connection error. Please try again.");
                btnSignInSubmit.disabled = false;
                signinEmail.disabled = false;
                signinPassword.disabled = false;
                signinBtnIcon.className = "fa-solid fa-arrow-right-to-bracket";
                signinBtnText.textContent = "Sign In to Dashboard";
            }
        });

        // Clear error states on input
        signinEmail.addEventListener("input", () => {
            if (signinEmail.classList.contains("is-invalid")) {
                signinEmail.classList.remove("is-invalid");
                signinEmailError.classList.remove("show");
            }
        });

        signinPassword.addEventListener("input", () => {
            if (signinPassword.classList.contains("is-invalid")) {
                signinPassword.classList.remove("is-invalid");
                signinPasswordError.classList.remove("show");
            }
        });
    }

    // =========================================================================
    // REGISTRATION SUBMISSION HANDLER
    // =========================================================================
    const formRegister = document.getElementById("pane-register");
    const btnRegisterSubmit = document.getElementById("btnRegisterSubmit");
    const regBtnIcon = document.getElementById("regBtnIcon");
    const regBtnText = document.getElementById("regBtnText");

    const regFullName = document.getElementById("regFullName");
    const regEmail = document.getElementById("regEmail");
    const regRole = document.getElementById("regRole");
    const regPassword = document.getElementById("regPassword");

    const regNameError = document.getElementById("regNameError");
    const regEmailError = document.getElementById("regEmailError");
    const regPasswordError = document.getElementById("regPasswordError");

    if (formRegister) {
        formRegister.addEventListener("submit", async (e) => {
            e.preventDefault();
            hideAlert();

            const fullName = (regFullName.value || "").trim();
            const email = (regEmail.value || "").trim();
            const role = regRole.value || "Inventory Analyst";
            const password = regPassword.value || "";

            let hasError = false;

            if (!fullName) {
                regFullName.classList.add("is-invalid");
                regNameError.classList.add("show");
                hasError = true;
            } else {
                regFullName.classList.remove("is-invalid");
                regNameError.classList.remove("show");
            }

            if (!email || !isValidEmail(email)) {
                regEmail.classList.add("is-invalid");
                regEmailError.classList.add("show");
                hasError = true;
            } else {
                regEmail.classList.remove("is-invalid");
                regEmailError.classList.remove("show");
            }

            if (!password || password.length < 6) {
                regPassword.classList.add("is-invalid");
                regPasswordError.classList.add("show");
                hasError = true;
            } else {
                regPassword.classList.remove("is-invalid");
                regPasswordError.classList.remove("show");
            }

            if (hasError) return;

            btnRegisterSubmit.disabled = true;
            regBtnIcon.className = "fa-solid fa-circle-notch fa-spin";
            regBtnText.textContent = "Creating Account...";

            try {
                const response = await fetch("/api/auth/register", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        full_name: fullName,
                        email: email,
                        role: role,
                        password: password
                    })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    showAlert("success", "Account created successfully! Redirecting to workspace...");
                    setTimeout(() => {
                        window.location.href = "/";
                    }, 800);
                } else {
                    showAlert("error", data.error || "Failed to create account.");
                    btnRegisterSubmit.disabled = false;
                    regBtnIcon.className = "fa-solid fa-user-plus";
                    regBtnText.textContent = "Create Enterprise Account";
                }
            } catch (err) {
                showAlert("error", "Network error while creating account.");
                btnRegisterSubmit.disabled = false;
                regBtnIcon.className = "fa-solid fa-user-plus";
                regBtnText.textContent = "Create Enterprise Account";
            }
        });
    }

    // =========================================================================
    // PASSWORD RECOVERY SUBMISSION HANDLER
    // =========================================================================
    const formRecover = document.getElementById("pane-recover");
    const btnRecoverSubmit = document.getElementById("btnRecoverSubmit");
    const recoverBtnIcon = document.getElementById("recoverBtnIcon");
    const recoverBtnText = document.getElementById("recoverBtnText");
    const recoverEmail = document.getElementById("recoverEmail");
    const recoverEmailError = document.getElementById("recoverEmailError");

    const tokenResetSection = document.getElementById("tokenResetSection");
    const resetToken = document.getElementById("resetToken");
    const resetNewPassword = document.getElementById("resetNewPassword");
    const btnApplyPasswordReset = document.getElementById("btnApplyPasswordReset");

    if (formRecover) {
        formRecover.addEventListener("submit", async (e) => {
            e.preventDefault();
            hideAlert();

            const email = (recoverEmail.value || "").trim();
            if (!email || !isValidEmail(email)) {
                recoverEmail.classList.add("is-invalid");
                recoverEmailError.classList.add("show");
                return;
            }

            recoverEmail.classList.remove("is-invalid");
            recoverEmailError.classList.remove("show");

            btnRecoverSubmit.disabled = true;
            recoverBtnIcon.className = "fa-solid fa-circle-notch fa-spin";
            recoverBtnText.textContent = "Generating Token...";

            try {
                const response = await fetch("/api/auth/forgot-password", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email: email })
                });

                const data = await response.json();
                btnRecoverSubmit.disabled = false;
                recoverBtnIcon.className = "fa-solid fa-paper-plane";
                recoverBtnText.textContent = "Send Recovery Request";

                if (response.ok && data.success) {
                    if (data.token) {
                        showAlert("success", "Security token generated for local/testing environment. Enter your new password below.");
                        if (tokenResetSection) {
                            tokenResetSection.style.display = "flex";
                            resetToken.value = data.token;
                        }
                    } else {
                        showAlert("info", data.message || "Recovery instructions sent.");
                    }
                } else {
                    showAlert("error", data.error || "Recovery request failed.");
                }
            } catch (err) {
                btnRecoverSubmit.disabled = false;
                recoverBtnIcon.className = "fa-solid fa-paper-plane";
                recoverBtnText.textContent = "Send Recovery Request";
                showAlert("error", "Network error processing recovery request.");
            }
        });

        if (btnApplyPasswordReset) {
            btnApplyPasswordReset.addEventListener("click", async () => {
                const token = (resetToken.value || "").trim();
                const newPw = resetNewPassword.value || "";

                if (!token || !newPw || newPw.length < 6) {
                    showAlert("error", "Token and new password (min 6 characters) are required.");
                    return;
                }

                btnApplyPasswordReset.disabled = true;
                btnApplyPasswordReset.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Updating...';

                try {
                    const response = await fetch("/api/auth/reset-password", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ token: token, new_password: newPw })
                    });
                    const data = await response.json();
                    btnApplyPasswordReset.disabled = false;
                    btnApplyPasswordReset.innerHTML = '<i class="fa-solid fa-check"></i> Set New Password';

                    if (response.ok && data.success) {
                        showAlert("success", data.message || "Password updated. You may now sign in.");
                        setTimeout(() => switchTab("pane-signin"), 1200);
                    } else {
                        showAlert("error", data.error || "Failed to update password.");
                    }
                } catch (err) {
                    btnApplyPasswordReset.disabled = false;
                    btnApplyPasswordReset.innerHTML = '<i class="fa-solid fa-check"></i> Set New Password';
                    showAlert("error", "Network error updating password.");
                }
            });
        }
    }
});
