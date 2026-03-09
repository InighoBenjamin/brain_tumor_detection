package com.braintumor.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "radiologist")
public class Radiologist {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer radiologistId;

    @Column(nullable = false, length = 100)
    private String radiologistName;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false, unique = true)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "lab_id", nullable = false)
    private Lab lab;

    @Column(updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();
}
