package com.braintumor.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "hospital")
public class Hospital {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer hospitalId;

    @Column(nullable = false, length = 150)
    private String hospitalName;

    @Column(length = 150)
    private String location;

    @Column(updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();
}
