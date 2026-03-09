package com.braintumor.controller;

import com.braintumor.config.JwtUtils;
import com.braintumor.entity.Role;
import com.braintumor.entity.User;
import com.braintumor.repository.UserRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.*;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authManager;
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtils jwtUtils;

    // ----------------------------------------------------------------
    // POST /api/auth/login
    // Body: { "email": "...", "password": "..." }
    // Returns: { "token": "eyJ...", "role": "doctor" }
    // ----------------------------------------------------------------
    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest req) {
        Authentication auth = authManager.authenticate(
            new UsernamePasswordAuthenticationToken(req.getEmail(), req.getPassword())
        );

        User user = userRepository.findByEmail(req.getEmail()).orElseThrow();
        String role = user.getRole().getRoleName().name();
        String token = jwtUtils.generateToken(req.getEmail(), role);

        return ResponseEntity.ok(Map.of(
            "token", token,
            "role",  role,
            "email", req.getEmail(),
            "userId", user.getUserId()
        ));
    }

    // ----------------------------------------------------------------
    // POST /api/auth/register
    // Body: { "userName": "...", "email": "...", "password": "...", "roleId": 5 }
    // Returns: 201 Created
    // ----------------------------------------------------------------
    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody RegisterRequest req) {
        if (userRepository.existsByEmail(req.getEmail())) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                .body(Map.of("error", "Email already registered"));
        }

        Role role = new Role();
        role.setRoleId(req.getRoleId());

        User user = new User();
        user.setUserName(req.getUserName());
        user.setEmail(req.getEmail());
        user.setPasswordHash(passwordEncoder.encode(req.getPassword()));
        user.setRole(role);

        userRepository.save(user);
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(Map.of("message", "User registered successfully"));
    }

    // ----------------------------------------------------------------
    // Request DTOs
    // ----------------------------------------------------------------
    @Data
    public static class LoginRequest {
        @Email @NotBlank
        private String email;
        @NotBlank
        private String password;
    }

    @Data
    public static class RegisterRequest {
        @NotBlank @Size(max = 100)
        private String userName;
        @Email @NotBlank
        private String email;
        @NotBlank @Size(min = 8, max = 100)
        private String password;
        private Integer roleId;
    }
}
