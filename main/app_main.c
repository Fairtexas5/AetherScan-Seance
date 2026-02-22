#include <stdio.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

// ── YOUR HOTSPOT CREDENTIALS ────────────────────────────────────
#define WIFI_SSID "ashes"
#define WIFI_PASS "987654321"
#define WIFI_CHANNEL 0  // 0 = auto-detect channel

static const char* TAG = "CSI_RX";
static bool wifi_connected = false;

// ── CSI CALLBACK: runs for every captured packet ────────────────
void csi_callback(void* ctx, wifi_csi_info_t* data) {
    if (!data || !data->buf) return;
    int8_t* buf = data->buf;
    int len = data->len;

    int n_subcarriers = len / 2;
    printf("CSI_DATA,%llu,%d,%d,", (unsigned long long)esp_timer_get_time(),
           data->rx_ctrl.rssi, n_subcarriers);

    for (int i = 0; i < len; i++) {
        printf("%d%s", buf[i], (i < len - 1) ? "," : "");
    }
    printf("\n");
}

// ── WIFI EVENT HANDLER ──────────────────────────────────────────
static void wifi_event_handler(void* arg, esp_event_base_t base, int32_t id,
                               void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START)
        esp_wifi_connect();
    else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "Connected to hotspot! CSI starting...");
        wifi_connected = true;
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_connected = false;
        ESP_LOGW(TAG, "Disconnected. Retrying...");
        esp_wifi_connect();
    }
}

// ── MAIN ENTRY POINT ───────────────────────────────────────────
void app_main(void) {
    ESP_LOGI(TAG, "=== ESP32 CSI Hotspot Receiver Starting ===");

    // Initialise NVS (required by WiFi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler,
                               NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                               wifi_event_handler, NULL);

    wifi_config_t sta_cfg = {.sta = {.ssid = WIFI_SSID,
                                     .password = WIFI_PASS,
                                     .channel = WIFI_CHANNEL}};
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);
    esp_wifi_start();
    esp_wifi_set_ps(WIFI_PS_NONE);

    ESP_LOGI(TAG, "Connecting to hotspot: %s ...", WIFI_SSID);

    // Wait up to 15 seconds for connection
    int retries = 0;
    while (!wifi_connected && retries < 30) {
        vTaskDelay(pdMS_TO_TICKS(500));
        retries++;
    }

    if (!wifi_connected) {
        ESP_LOGE(TAG,
                 "FAILED TO CONNECT. Check SSID/password and hotspot is ON.");
        return;
    }

    // ── Enable CSI capture ────────────────────────────────────
    wifi_csi_config_t csi_cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };
    esp_wifi_set_csi_config(&csi_cfg);
    esp_wifi_set_csi_rx_cb(csi_callback, NULL);
    esp_wifi_set_csi(true);

    ESP_LOGI(TAG, "CSI active. Packets streaming to Serial Monitor...");
    ESP_LOGI(TAG,
             "Format: CSI_DATA, timestamp_us, rssi_dBm, n_subcarriers, IQ...");

    // Keep running forever
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
