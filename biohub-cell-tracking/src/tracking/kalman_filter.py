"""Kalman filter for motion prediction in cell tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """State of Kalman filter."""
    position: np.ndarray  # (z, y, x)
    velocity: np.ndarray  # (vz, vy, vx)
    covariance: np.ndarray  # 6x6 covariance matrix


class KalmanFilter3D:
    """Kalman filter for 3D cell motion prediction."""
    
    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        initial_velocity_std: float = 1.0
    ):
        """Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            initial_velocity_std: Initial uncertainty in velocity
        """
        # State vector: [z, y, x, vz, vy, vx]
        self.state_dim = 6
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        self.F[0, 3] = 1.0  # z += vz
        self.F[1, 4] = 1.0  # y += vy
        self.F[2, 5] = 1.0  # x += vx
        
        # Measurement matrix (observe position only)
        self.H = np.zeros((3, self.state_dim))
        self.H[0, 0] = 1.0  # observe z
        self.H[1, 1] = 1.0  # observe y
        self.H[2, 2] = 1.0  # observe x
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # Initial state covariance
        self.P0 = np.eye(self.state_dim)
        self.P0[3:, 3:] = np.eye(3) * initial_velocity_std**2
        
        self.state: Optional[KalmanState] = None
        
        logger.info(
            f"KalmanFilter3D initialized: process_noise={process_noise}, "
            f"measurement_noise={measurement_noise}"
        )
    
    def initialize(self, position: np.ndarray, velocity: Optional[np.ndarray] = None) -> None:
        """Initialize filter with first measurement.
        
        Args:
            position: Initial position (z, y, x)
            velocity: Initial velocity (vz, vy, vx), defaults to zeros
        """
        if velocity is None:
            velocity = np.zeros(3)
        
        state_vector = np.concatenate([position, velocity])
        self.state = KalmanState(
            position=position,
            velocity=velocity,
            covariance=self.P0.copy()
        )
    
    def predict(self, dt: float = 1.0) -> KalmanState:
        """Predict next state.
        
        Args:
            dt: Time step (default 1.0)
        
        Returns:
            Predicted state
        """
        if self.state is None:
            raise ValueError("Kalman filter not initialized")
        
        # Update state transition for time step
        F_dt = self.F.copy()
        F_dt[0, 3] = dt
        F_dt[1, 4] = dt
        F_dt[2, 5] = dt
        
        # Predict state
        state_vector = np.concatenate([self.state.position, self.state.velocity])
        predicted_state = F_dt @ state_vector
        
        # Predict covariance
        predicted_covariance = F_dt @ self.state.covariance @ F_dt.T + self.Q * dt
        
        predicted_position = predicted_state[:3]
        predicted_velocity = predicted_state[3:]
        
        return KalmanState(
            position=predicted_position,
            velocity=predicted_velocity,
            covariance=predicted_covariance
        )
    
    def update(self, measurement: np.ndarray) -> KalmanState:
        """Update filter with new measurement.
        
        Args:
            measurement: Measured position (z, y, x)
        
        Returns:
            Updated state
        """
        if self.state is None:
            raise ValueError("Kalman filter not initialized")
        
        # Predict step
        predicted = self.predict()
        
        # Compute Kalman gain
        S = self.H @ predicted.covariance @ self.H.T + self.R
        K = predicted.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        state_vector = np.concatenate([predicted.position, predicted.velocity])
        innovation = measurement - self.H @ state_vector[:3]
        updated_state = state_vector + K @ innovation
        
        # Update covariance
        I = np.eye(self.state_dim)
        updated_covariance = (I - K @ self.H) @ predicted.covariance
        
        self.state = KalmanState(
            position=updated_state[:3],
            velocity=updated_state[3:],
            covariance=updated_covariance
        )
        
        return self.state
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        if self.state is None:
            raise ValueError("Kalman filter not initialized")
        return self.state.position
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        if self.state is None:
            raise ValueError("Kalman filter not initialized")
        return self.state.velocity
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance)."""
        if self.state is None:
            raise ValueError("Kalman filter not initialized")
        return np.trace(self.state.covariance[:3, :3])
