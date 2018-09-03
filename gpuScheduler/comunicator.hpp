#ifndef _COMUNICATOR_NOT_INCLUDED_
#define _COMUNICATOR_NOT_INCLUDED_

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdint>

#include <amqp_tcp_socket.h>
#include <amqp.h>
#include "utils.h"

class Comunicator {
private:
char const* exchange;
char const* hostname;
char const* queue_name;
static const int fixed_channel_id = 1;
int port, status;
amqp_socket_t *socket;
amqp_connection_state_t conn;
amqp_bytes_t queuename;
public:
Comunicator(char const* hostname="localhost",int port=5672,char const* queue_name="test_scheduler",char const* exchange="amq.direct"){
	this->hostname = hostname;
	this->port = port;
	this->socket = NULL;
	this->queue_name=queue_name;
}

void setupConnection(){
	this->conn = amqp_new_connection();
	this->socket = amqp_tcp_socket_new(this->conn);
	if(!this->socket) die("creating TCP socket");

	this->status = amqp_socket_open(this->socket, this->hostname, this->port);
	if(this->status) die("opening TCP socket");

	die_on_amqp_error(amqp_login(this->conn,"/",1,AMQP_DEFAULT_FRAME_SIZE,AMQP_DEFAULT_HEARTBEAT,AMQP_SASL_METHOD_PLAIN,"guest","guest"),"Logging in");

	amqp_channel_open(this->conn, fixed_channel_id);
	queue_declare();
}

void closeConnection() {
	amqp_rpc_reply_t rpc_reply =
		amqp_connection_close(this->conn, AMQP_REPLY_SUCCESS);
	assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

	int rc = amqp_destroy_connection(this->conn);
	assert(rc == AMQP_STATUS_OK);
}

void queue_declare() {
	amqp_queue_declare_ok_t *res = amqp_queue_declare(
		this->conn, this->fixed_channel_id, amqp_cstring_bytes(this->queue_name),
		/*passive*/ 0,
		/*durable*/ 0,
		/*exclusive*/ 0,
		/*auto_delete*/ 1, amqp_empty_table);
	assert(res != NULL);
}

char *getMessage() {
	amqp_basic_consume_ok_t *result =
		amqp_basic_consume(this->conn, this->fixed_channel_id,
		                   amqp_cstring_bytes(this->queue_name), amqp_empty_bytes,
		                   /*no_local*/ 0,
		                   /*no_ack*/ 1,
		                   /*exclusive*/ 0, amqp_empty_table);
	assert(result != NULL);

	amqp_envelope_t envelope;
	struct timeval timeout = {5, 0};
	amqp_rpc_reply_t rpc_reply =
		amqp_consume_message(this->conn, &envelope, &timeout, 0);
	assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

	uint64_t out_body_size_ = envelope.message.body.len;
	char *body = (char *)malloc(sizeof(char)*out_body_size_);
	if (out_body_size_) {
		memcpy(body, envelope.message.body.bytes, out_body_size_);
	}
	amqp_destroy_envelope(&envelope);
	return body;
}

amqp_connection_state_t setup_connection_and_channel(void) {
	amqp_connection_state_t connection_state_ = amqp_new_connection();

	amqp_socket_t *socket = amqp_tcp_socket_new(connection_state_);
	assert(socket);

	int rc = amqp_socket_open(socket, "localhost", AMQP_PROTOCOL_PORT);
	assert(rc == AMQP_STATUS_OK);

	amqp_rpc_reply_t rpc_reply = amqp_login(
		connection_state_, "/", 1, AMQP_DEFAULT_FRAME_SIZE,
		AMQP_DEFAULT_HEARTBEAT, AMQP_SASL_METHOD_PLAIN, "guest", "guest");
	assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

	amqp_channel_open_ok_t *res =
		amqp_channel_open(connection_state_, fixed_channel_id);
	assert(res != NULL);

	return connection_state_;
}

void close_and_destroy_connection(amqp_connection_state_t connection_state_) {
	amqp_rpc_reply_t rpc_reply =
		amqp_connection_close(connection_state_, AMQP_REPLY_SUCCESS);
	assert(rpc_reply.reply_type == AMQP_RESPONSE_NORMAL);

	int rc = amqp_destroy_connection(connection_state_);
	assert(rc == AMQP_STATUS_OK);
}

};
#endif
