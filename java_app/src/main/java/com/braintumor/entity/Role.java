package com.braintumor.entity;

import jakarta.persistence.*;
import lombok.Data;

@Data
@Entity
@Table(name = "role")
public class Role {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer roleId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private RoleName roleName;

    public enum RoleName {
        admin, doctor, radiologist, lab_staff, patient
    }
}
